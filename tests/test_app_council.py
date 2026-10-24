import io
import json
import os
import tempfile
import threading
import time
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault('gpt_api_key', 'test-placeholder-no-network')
import main
from GPT_handle import _build_user_content
from services.config_store import ConfigStore, DEFAULT_COUNCIL_CONFIG
from services.council_workflow import CouncilStopped


class CouncilAppTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)
        config = deepcopy(DEFAULT_COUNCIL_CONFIG)
        config['memory']['enabled'] = False
        self.root.joinpath('config.json').write_text(json.dumps(config))
        self.store = patch.object(main, 'config_store', ConfigStore(str(self.root)))
        self.store.start()
        self.app_config = patch.dict(main.app.config, {
            key: str(self.root / name) for key, name in (
                ('UPLOAD_FOLDER', 'uploads'), ('CHAT_HISTORY_FOLDER', 'chats'),
                ('TEMP_CHAT_HISTORY_FOLDER', 'temp'))})
        self.app_config.start()
        for key in ('UPLOAD_FOLDER', 'CHAT_HISTORY_FOLDER', 'TEMP_CHAT_HISTORY_FOLDER'):
            Path(main.app.config[key]).mkdir()
        main.conversations.clear()
        self.client = main.app.test_client()
        self.calls = []
        self.transport = patch.object(main, 'completion_response_stream', self.stream)
        self.transport.start()
        self.outgoing_patch = patch.object(main.socketio, "emit", wraps=main.socketio.emit)
        self.outgoing = self.outgoing_patch.start()

    def tearDown(self):
        self.outgoing_patch.stop()
        self.transport.stop()
        self.app_config.stop()
        self.store.stop()
        self.folder.cleanup()

    def stream(self, **kwargs):
        self.calls.append(kwargs)
        prompt = kwargs['system_prompt']
        if 'Make your own complete proposal' in prompt:
            text = 'Independent complete answer.'
        elif 'single scheduled turn' in prompt:
            text = json.dumps({'contribution': 'Substantive discussion.', 'needs_more_discussion': False})
        elif 'You are the chair, now integrating' in prompt:
            text = 'Exact approved answer.\n\nEvidence and uncertainty.'
        elif 'Review the exact chair draft independently' in prompt:
            text = '{"vote":"approve","reason":"Checked against the source."}'
        else:
            raise AssertionError('Unexpected helper-model request')
        yield 'content', text[:10]
        yield 'content', text[10:]

    def test_pdf_upload_and_restore_do_not_invoke_models(self):
        response = self.client.post('/api/upload_document', data={
            'chat_id': 'test', 'file': (io.BytesIO(b'%PDF-1.4 test source'), 'source.pdf')})
        data = response.get_json()
        self.assertTrue(data['success'])
        self.assertNotIn('pdf_brief_summary', data['document'])
        restored = self.client.post('/api/restore_documents', json={'chat_id': 'other', 'documents': [data['document']]})
        self.assertTrue(restored.get_json()['success'])
        self.assertEqual(self.calls, [])
        system, images, pdfs = main.build_document_context('other', 'S')
        self.assertIn('source.pdf', system)
        blocks = _build_user_content('Q', images, pdfs)
        self.assertEqual(blocks[-1]['file']['filename'], 'source.pdf')
        self.assertTrue(blocks[-1]['file']['file_data'].startswith('data:application/pdf;base64,'))

    def test_full_socket_workflow_direct_files_and_persistent_ballot_metadata(self):
        self.client.post('/api/upload_document', data={
            'chat_id': 'sitting', 'file': (io.BytesIO(b'%PDF-1.4 test source'), 'source.pdf')})
        socket = main.socketio.test_client(main.app, flask_test_client=self.client)
        try:
            socket.emit('join_chat', {'chat_id': 'sitting'})
            socket.emit('send_message', {'chat_id': 'sitting', 'message': 'Solve this using the PDF'})
            events = []
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                # Flask-SocketIO 5.3's test client misses optimized Engine.IO
                # packets in python-socketio 5.10. Inspect real outgoing calls.
                events = [{'name': call.args[0], 'args': [call.args[1]]}
                          for call in list(self.outgoing.call_args_list)]
                if any(event['name'] == 'all_done' for event in events):
                    break
                time.sleep(0.01)
            self.assertTrue(any(event['name'] == 'all_done' for event in events))
            self.assertEqual(len(self.calls), 9)  # 3 proposals + 3 discussion + draft + 2 ballots
            self.assertTrue(all(call['pdf_inputs'][0]['filename'] == 'source.pdf' for call in self.calls))
            self.assertTrue(all(call['timeout_seconds'] == 120 for call in self.calls))
            chunks = [event['args'][0] for event in events if event['name'] == 'ai_response_chunk']
            self.assertFalse(any(chunk.get('council_phase') == 'vote' for chunk in chunks))
            finals = [event['args'][0] for event in events if event['name'] == 'ai_response_end' and event['args'][0].get('is_final_response')]
            self.assertEqual(len(finals), 1)
            self.assertTrue(finals[0]['council_payload']['approved'])
            self.assertIn('Exact approved answer.', finals[0]['message'])
            persisted = json.loads((self.root / 'temp/sitting.json').read_text())
            self.assertEqual(persisted['schema_version'], 4)
            votes = [msg for msg in persisted['messages'] if msg.get('council_phase') == 'vote']
            self.assertEqual(len(votes), 2)
            self.assertTrue(all(msg['council_payload']['vote'] == 'approve' for msg in votes))
            main._replace_conversation_messages_from_ui('roundtrip', persisted['messages'])
            rebuilt = main._history_to_ui_messages('roundtrip')
            self.assertEqual([m['council_payload'] for m in rebuilt if m.get('council_phase') == 'vote'],
                             [m['council_payload'] for m in votes])
            history = main._build_prompt_chat_history(main.get_conversation('roundtrip')['messages'], len(rebuilt), 'final_only')
            self.assertEqual(len(history), 2)
            self.assertFalse(main.get_conversation('sitting')['is_generating'])
        finally:
            socket.disconnect()

    def test_timeout_discards_late_worker_and_next_turn_runs(self):
        late_finished = threading.Event()
        def slow(**kwargs):
            time.sleep(0.08)
            yield 'content', 'LATE MUST NOT APPEAR'
            late_finished.set()
        conv = main.get_conversation('timeout')
        conv['is_generating'] = True
        emitted = []
        kwargs = dict(conversation_id='timeout', emit_chat=lambda *args: emitted.append(args),
                      member={'id': 'member', 'name': 'Member', 'model': 'm'}, system='S', prompt='Q',
                      phase='proposal', number=0, group='group', history=[], documents={},
                      timeout=0.02, aborted=conv['abort_event'].is_set)
        with patch.object(main, 'completion_response_stream', slow):
            with self.assertRaises(TimeoutError): main._request_council_turn(**kwargs)
        before = deepcopy(conv['messages'])
        time.sleep(0.1)
        self.assertEqual(conv['messages'], before)
        self.assertFalse(any(event == 'ai_response_chunk' for event, _ in emitted))
        self.assertIsNone(conv['pending_message_id'])
        kwargs['timeout'] = 1
        kwargs['system'] = 'Make your own complete proposal'
        self.assertIn('Independent', main._request_council_turn(**kwargs))

    def test_cancel_stops_controller_without_final_answer(self):
        conv = main.get_conversation('cancel')
        def stopped_stream(**kwargs):
            conv['abort_event'].set()
            yield 'content', 'discard'
        with patch.object(main, 'completion_response_stream', stopped_stream):
            with self.assertRaises(CouncilStopped):
                main._request_council_turn(
                    conversation_id='cancel', emit_chat=lambda *args: None,
                    member={'id': 'member', 'name': 'Member', 'model': 'm'}, system='S', prompt='Q',
                    phase='proposal', number=0, group='g', history=[], documents={}, timeout=1,
                    aborted=conv['abort_event'].is_set)
        self.assertEqual(conv['messages'][0]['stream_status'], 'stopped')
        self.assertFalse(any(msg.get('is_final_response') for msg in conv['messages']))

    def test_council_api_and_manual_memory(self):
        roster = self.client.get('/api/council').get_json()
        self.assertEqual(list(roster), ['chair', 'member-1', 'member-2'])
        self.assertTrue(roster['chair']['is_chair'])
        response = self.client.get('/api/memories').get_json()
        self.assertFalse(response['auto_extract'])

    def test_legacy_final_remains_in_history(self):
        messages = [{'type': 'user', 'content': 'Q'},
                    {'type': 'ai', 'sender': 'Researcher', 'content': 'old intermediate'},
                    {'type': 'ai', 'sender': 'Leader - Final Response (model)', 'content': 'old final'}]
        main._replace_conversation_messages_from_ui('legacy', messages)
        history = main._build_prompt_chat_history(main.get_conversation('legacy')['messages'], 3, 'final_only')
        self.assertEqual([message['content'] for message in history], ['Q', 'old final'])


if __name__ == '__main__':
    unittest.main()
