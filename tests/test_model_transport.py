import os
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

os.environ.setdefault('gpt_api_key', 'test-placeholder-no-network')
import GPT_handle


class TransportTests(unittest.TestCase):
    def test_council_timeout_disables_sdk_retries_and_closes_stream(self):
        response = MagicMock()
        response.__iter__.return_value = iter([
            SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content='answer'))])])
        request_client = MagicMock()
        request_client.chat.completions.create.return_value = response
        with patch.object(GPT_handle.client, 'with_options', return_value=request_client) as options, \
             patch.object(GPT_handle, 'log_info'):
            chunks = list(GPT_handle.completion_response_stream('provider/model', 'S', 'Q', timeout_seconds=7))
        options.assert_called_once_with(timeout=7, max_retries=0)
        self.assertEqual(chunks, [('content', 'answer')])
        response.close.assert_called_once()

    def test_cancelled_transport_makes_no_request(self):
        cancelled = threading.Event(); cancelled.set()
        with patch.object(GPT_handle.client, 'with_options') as options:
            chunks = list(GPT_handle.completion_response_stream('m', 'S', 'Q', timeout_seconds=7, cancel_event=cancelled))
        self.assertEqual(chunks, [])
        options.return_value.chat.completions.create.assert_not_called()

    def test_missing_pdf_is_not_silently_dropped(self):
        with self.assertRaises(ValueError):
            GPT_handle._build_user_content('Q', pdf_inputs=[{'filename':'missing.pdf','filepath':'/nonexistent-council-test.pdf'}])


if __name__ == '__main__':
    unittest.main()
