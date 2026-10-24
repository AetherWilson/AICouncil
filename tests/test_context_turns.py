import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault('gpt_api_key', 'test-placeholder-no-network')
import main
from services.context_manager import ContextManager, IncompleteResponseError


class ContextTurnTests(unittest.TestCase):
    def setUp(self):
        main.conversations.clear()
        self.events = []
        self.save = patch.object(main, 'persist_chat_snapshot'); self.save.start()
        self.encoding = patch('services.context_manager._encoding', return_value=None); self.encoding.start()
        self.conv = main.get_conversation('test-context')
        self.kwargs = dict(conversation_id='test-context', emit_chat=lambda *args: self.events.append(args),
                           member={'id': 'seat', 'name': 'Seat', 'model': 'scheduled-model'}, system='Canonical instructions',
                           prompt=json.dumps({'user_request': 'Exact task', 'council_record': []}),
                           phase='draft', number=1, group='g', history=[], documents={}, timeout=2,
                           aborted=self.conv['abort_event'].is_set)

    def tearDown(self):
        self.encoding.stop(); self.save.stop()

    def test_long_response_continues_same_seat_without_duplicate_text(self):
        calls = []
        def stream(**kwargs):
            calls.append(kwargs)
            if not kwargs.get('prefix'):
                yield 'content', 'First part. '
                yield 'finish', 'length'
            else:
                self.assertEqual(kwargs['prefix'], 'First part. ')
                yield 'content', 'Second part.'
                yield 'finish', 'stop'
        with patch.object(main, 'completion_response_stream', stream):
            result = main._request_council_turn(**self.kwargs)
        self.assertEqual(result, 'First part. Second part.')
        self.assertEqual(self.conv['messages'][0]['raw_markdown'], result)
        self.assertEqual([call['model'] for call in calls], ['scheduled-model', 'scheduled-model'])
        self.assertEqual(len([e for e, _ in self.events if e == 'ai_response_start']), 1)
        self.assertTrue(any(event == 'context_status' and data['status'] == 'continuing' for event, data in self.events))

    def test_exhausted_continuations_remain_incomplete(self):
        def stream(**kwargs):
            yield 'content', 'partial'
            yield 'finish', 'length'
        manager = ContextManager(settings={'max_continuations': 0}, history=[], summarize=lambda *args: '')
        with patch.object(main, 'completion_response_stream', stream):
            with self.assertRaises(IncompleteResponseError):
                main._request_council_turn(**self.kwargs, context_manager=manager)
        self.assertEqual(self.conv['messages'][0]['stream_status'], 'error')
        self.assertFalse(self.conv['messages'][0]['is_final_response'])

    def test_overflow_compacts_with_same_model_and_retries_once(self):
        calls, attempts = [], [0]
        history = [{'role': 'assistant', 'content': 'Old findings and decisions. ' * 100}]
        manager = ContextManager(settings={}, history=history, summarize=lambda *args: '')
        def stream(**kwargs):
            calls.append(kwargs)
            if 'Create a concise context checkpoint' in kwargs['system_prompt']:
                yield 'content', 'Goal: preserve constraints. Evidence: earlier findings. Next: finish.'
                yield 'finish', 'stop'
            elif attempts[0] == 0:
                attempts[0] += 1
                raise RuntimeError('context_length_exceeded')
            else:
                yield 'content', 'Recovered complete answer.'
                yield 'finish', 'stop'
        with patch.object(main, 'completion_response_stream', stream):
            result = main._request_council_turn(**self.kwargs, context_manager=manager)
        self.assertEqual(result, 'Recovered complete answer.')
        self.assertEqual(len(calls), 3)
        self.assertTrue(all(call['model'] == 'scheduled-model' for call in calls))
        self.assertTrue(manager.history_checkpoint)
        self.assertTrue(any('Context checkpoint' in item['content'] for item in calls[-1]['chat_history']))

    def test_repeated_overflow_is_bounded(self):
        count = [0]
        def stream(**kwargs):
            count[0] += 1
            raise RuntimeError('maximum context length exceeded')
            yield
        with patch.object(main, 'completion_response_stream', stream):
            with self.assertRaises(RuntimeError):
                main._request_council_turn(**self.kwargs)
        self.assertEqual(count[0], 2)

    def test_private_ballot_continuation_never_streams_votes(self):
        calls = [0]
        def stream(**kwargs):
            calls[0] += 1
            if calls[0] == 1:
                yield 'content', '{"vote":"approve",'
                yield 'finish', 'length'
            else:
                yield 'content', '"reason":"Reviewed."}'
                yield 'finish', 'stop'
        kwargs = dict(self.kwargs, phase='vote')
        with patch.object(main, 'completion_response_stream', stream):
            raw = main._request_council_turn(**kwargs)
        self.assertEqual(json.loads(raw)['vote'], 'approve')
        self.assertFalse(any(event.startswith('ai_response') for event, _ in self.events))
        self.assertEqual(self.conv['messages'], [])

    def test_failed_snapshot_write_retains_previous_complete_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder, 'chat.json'))
            main._write_chat_file(path, {'context_state': {'summary': 'last good checkpoint'}})
            def interrupted(payload, handle, **kwargs):
                handle.write('{broken')
                raise OSError('simulated interrupted write')
            with patch.object(main.json, 'dump', side_effect=interrupted):
                with self.assertRaises(OSError):
                    main._write_chat_file(path, {'context_state': {'summary': 'new checkpoint'}})
            self.assertEqual(json.loads(Path(path).read_text())['context_state']['summary'], 'last good checkpoint')
            self.assertEqual(list(Path(folder).glob('*.tmp')), [])

    def test_context_checkpoint_survives_save_load_and_client_resave(self):
        self.save.stop()
        with tempfile.TemporaryDirectory() as folder:
            for name in ('saved', 'temp'): Path(folder, name).mkdir()
            with patch.dict(main.app.config, CHAT_HISTORY_FOLDER=str(Path(folder, 'saved')), TEMP_CHAT_HISTORY_FOLDER=str(Path(folder, 'temp'))):
                conv = main.get_conversation('chat_checkpoint')
                conv['context_state'] = {'version': 1, 'history_checkpoint': {'summary': 'Goal and decisions', 'covered_count': 2, 'prefix_hash': 'hash'}, 'compactions': 3}
                main.append_conversation_message('chat_checkpoint', 'user', 'Original question')
                main.persist_chat_snapshot('chat_checkpoint')
                original = copy.deepcopy(conv['context_state'])
                main.conversations.pop('chat_checkpoint')
                client = main.app.test_client()
                loaded = client.get('/api/load_chat/chat_checkpoint').get_json()
                self.assertEqual(main.get_conversation('chat_checkpoint')['context_state'], original)
                client.post('/api/save_chat', json={'id': 'chat_checkpoint', 'name': 'Renamed', 'messages': loaded['chat']['messages']})
                persisted = json.loads(Path(folder, 'saved/chat_checkpoint.json').read_text())
                self.assertEqual(persisted['context_state'], original)
                self.assertEqual(persisted['messages'][0]['content'], 'Original question')
        self.save.start()


if __name__ == '__main__':
    unittest.main()
