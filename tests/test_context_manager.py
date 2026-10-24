import copy
import json
import unittest
from unittest.mock import patch

from services.context_manager import (ContextLimitError, ContextManager, checkpoint_valid,
                                      digest, is_context_overflow, normalize_context, token_count)
from services.council_workflow import CouncilStopped


class ContextManagerTests(unittest.TestCase):
    def setUp(self):
        self.encoding = patch('services.context_manager._encoding', return_value=None)
        self.encoding.start()
        self.settings = {'context_window_tokens': 4096, 'reserve_output_tokens': 512,
                         'summary_tokens': 128, 'recent_history_tokens': 300,
                         'recent_council_records': 1}
        self.summaries, self.events = [], []

    def tearDown(self):
        self.encoding.stop()

    def summarize(self, model, system, source, output):
        self.summaries.append((model, json.loads(source), output))
        return 'Goal: retain the user deadline and constraints. Evidence: source.pdf. Open issue: verification. Next: continue the task.'

    def manager(self, history=None, **kwargs):
        return ContextManager(settings=self.settings, history=history or [], summarize=self.summarize,
                              notify=self.events.append, **kwargs)

    def prompt(self, records=None):
        return json.dumps({'user_request': 'CURRENT EXACT REQUEST: finish the work',
                           'user_updates': ['Keep the original constraints'], 'council_record': records or []})

    def test_small_requests_do_not_invoke_compaction(self):
        history = [{'role': 'user', 'content': 'Earlier question'}, {'role': 'assistant', 'content': 'Earlier answer'}]
        manager = self.manager(history)
        prepared = manager.prepare(model='same-model', system='Canonical instructions', prompt=self.prompt(), phase='proposal')
        self.assertEqual(prepared['history'], history)
        self.assertEqual(self.summaries, [])
        self.assertEqual(json.loads(prepared['prompt'])['user_request'], 'CURRENT EXACT REQUEST: finish the work')

    def test_large_history_checkpoint_preserves_originals_and_recent_user_input(self):
        history = [{'role': 'user' if i % 2 == 0 else 'assistant', 'content': f'entry-{i}: ' + 'detail ' * 350} for i in range(12)]
        original = copy.deepcopy(history)
        manager = self.manager(history)
        prepared = manager.prepare(model='scheduled-model', system='Canonical', prompt=self.prompt(), phase='proposal')
        self.assertTrue(self.summaries)
        self.assertTrue(all(model == 'scheduled-model' for model, *_ in self.summaries))
        self.assertEqual(history, original)
        self.assertLess(prepared['input_tokens'], 4096 - 512)
        self.assertTrue(checkpoint_valid(manager.history_checkpoint, history))
        self.assertIn(history[-2], prepared['history'])
        self.assertTrue(any('Context checkpoint' in item['content'] for item in prepared['history']))
        self.assertTrue(any(event['status'] == 'compacted' for event in self.events))

    def test_checkpoint_reuses_after_reload_and_invalidates_after_edit(self):
        history = [{'role': 'assistant', 'content': 'old work ' * 3000}]
        manager = self.manager(history)
        manager.prepare(model='m', system='S', prompt=self.prompt(), phase='proposal')
        saved = json.loads(json.dumps(manager.state()))
        self.summaries.clear()
        reloaded = self.manager(history, state=saved)
        reloaded.prepare(model='m', system='S', prompt=self.prompt(), phase='proposal')
        self.assertEqual(self.summaries, [])
        changed = copy.deepcopy(history); changed[0]['content'] = 'edited'
        edited = self.manager(changed, state=saved)
        self.assertEqual(edited.history_checkpoint, {})
        self.assertFalse(checkpoint_valid(saved['history_checkpoint'], changed))

    def test_compaction_keeps_current_draft_and_instructions_verbatim(self):
        records = [{'phase': 'discussion', 'member': f'm-{i}', 'text': 'evidence ' * 500} for i in range(6)]
        draft = {'phase': 'draft', 'text': 'Exact draft. Include this clause and no substitutions.'}
        records.append(draft)
        manager = self.manager()
        prepared = manager.prepare(model='reviewer', system='S', prompt=self.prompt(records), phase='vote')
        payload = json.loads(prepared['prompt'])
        self.assertEqual(payload['council_record'][-1], draft)
        self.assertEqual(payload['user_request'], json.loads(self.prompt())['user_request'])
        self.assertEqual(payload['user_updates'], ['Keep the original constraints'])
        self.assertLess(len(payload['council_record']), len(records))
        self.assertEqual(records[-1], draft)

    def test_oversize_protected_draft_or_attachment_fails_without_truncation(self):
        for records, attachments in ([{'phase': 'draft', 'text': 'MUST KEEP ' * 3000}], 0), ([], 5000):
            with self.subTest(attachments=attachments):
                manager = self.manager()
                original = copy.deepcopy(records)
                with self.assertRaises(ContextLimitError):
                    manager.prepare(model='reviewer', system='S', prompt=self.prompt(records), phase='vote', attachment_tokens=attachments)
                self.assertEqual(records, original)
                self.assertEqual(manager.record_checkpoint, {})

    def test_summary_failure_rolls_back_checkpoint(self):
        history = [{'role': 'assistant', 'content': 'old work ' * 3000}]
        manager = self.manager(history)
        def fail(*args):
            raise RuntimeError('summary failed')
        manager.summarize = fail
        with self.assertRaisesRegex(RuntimeError, 'summary failed'):
            manager.prepare(model='m', system='S', prompt=self.prompt(), phase='proposal')
        self.assertEqual(manager.history_checkpoint, {})
        self.assertEqual(manager.compactions, 0)

    def test_cancellation_during_summary_does_not_commit(self):
        cancelled = [False]
        manager = self.manager([{'role': 'assistant', 'content': 'old work ' * 3000}], stopped=lambda: cancelled[0])
        def cancel(*args):
            cancelled[0] = True
            return 'checkpoint'
        manager.summarize = cancel
        with self.assertRaises(CouncilStopped):
            manager.prepare(model='m', system='S', prompt=self.prompt(), phase='proposal')
        self.assertEqual(manager.history_checkpoint, {})

    def test_chunked_summary_processes_every_unicode_character(self):
        manager = self.manager()
        text = ('條件：保留檔案來源 source.pdf。\n' * 500)
        manager._summarize('m', text)
        self.assertGreater(len(self.summaries), 1)
        self.assertEqual(''.join(source['new_fragment'] for _, source, _ in self.summaries), text)
        self.assertTrue(all(source['previous_checkpoint'] for _, source, _ in self.summaries[1:]))

    def test_long_partial_answer_uses_checkpoint_and_exact_tail(self):
        manager = self.manager()
        partial = 'Already emitted and preserved.\n' * 1200
        prepared = manager.prepare(model='same-member', system='S', prompt=self.prompt(), phase='draft', partial_output=partial)
        self.assertTrue(partial.endswith(prepared['prefix']))
        self.assertLess(len(prepared['prefix']), len(partial))
        self.assertIn('Partial answer already emitted', prepared['history'][-1]['content'])
        self.assertEqual(''.join(source['new_fragment'] for _, source, _ in self.summaries), partial[:-len(prepared['prefix'])])

    def test_model_limits_and_common_window_are_respected(self):
        manager = self.manager(shared_window=2048, model_info=lambda model: {'context_window_tokens': 100000, 'max_output_tokens': 128})
        self.assertEqual(manager.limits('m'), (2048, 128, 1536))

    def test_tokenizer_unavailable_falls_back_without_discarding_data(self):
        self.assertGreater(token_count('繁體中文 and code'), 0)
        self.assertNotEqual(digest([{'content': 'a'}]), digest([{'content': 'b'}]))

    def test_context_overflow_is_detected_through_transport_wrapper(self):
        try:
            try:
                raise ValueError('context_length_exceeded')
            except ValueError as cause:
                raise RuntimeError('Model stream failed') from cause
        except RuntimeError as error:
            self.assertTrue(is_context_overflow(error))
        self.assertFalse(is_context_overflow(RuntimeError('401 invalid API key')))
        self.assertFalse(is_context_overflow(RuntimeError('rate_limit_exceeded: too many tokens per minute')))

    def test_bad_context_settings_fail_clearly(self):
        with self.assertRaises(ValueError): normalize_context({'summary_tokens': 'not a number'})


if __name__ == '__main__':
    unittest.main()
