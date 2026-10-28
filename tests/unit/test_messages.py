import io
import os
import unittest
import unittest.mock

import dgenerate.hfhub as hfhub
import dgenerate.messages as messages


class TestCensorSecrets(unittest.TestCase):

    def test_redacts_environment_tokens(self):
        previous_hf = os.environ.get('HF_TOKEN')
        previous_civit = os.environ.get('CIVIT_AI_TOKEN')
        os.environ['HF_TOKEN'] = 'hf_secret_token_value'
        os.environ['CIVIT_AI_TOKEN'] = 'civit_secret_token_value'
        try:
            text = (
                'hf_secret_token_value and civit_secret_token_value '
                '\\env "HF_TOKEN=hf_secret_token_value"')
            censored = messages.censor_secrets(text)
            self.assertNotIn('hf_secret_token_value', censored)
            self.assertNotIn('civit_secret_token_value', censored)
            self.assertIn('HF_TOKEN=***', censored)
        finally:
            if previous_hf is None:
                os.environ.pop('HF_TOKEN', None)
            else:
                os.environ['HF_TOKEN'] = previous_hf
            if previous_civit is None:
                os.environ.pop('CIVIT_AI_TOKEN', None)
            else:
                os.environ['CIVIT_AI_TOKEN'] = previous_civit

    def test_redacts_auth_token_context(self):
        previous = os.environ.pop('HF_TOKEN', None)
        try:
            with hfhub.auth_token_context('session_only_hf_token'):
                censored = messages.censor_secrets('token session_only_hf_token')
            self.assertEqual(censored, 'token ***')
        finally:
            if previous is not None:
                os.environ['HF_TOKEN'] = previous

    def test_redacts_civitai_download_query_token(self):
        url = 'https://civitai.com/api/download/models/1?token=leaked_query_token'
        censored = messages.censor_secrets(url)
        self.assertIn('token=***', censored)
        self.assertNotIn('leaked_query_token', censored)

    def test_log_output_is_censored(self):
        previous = os.environ.get('HF_TOKEN')
        os.environ['HF_TOKEN'] = 'logged_hf_token_value'
        buffer = io.StringIO()
        previous_file = messages.get_message_file()
        try:
            messages.set_message_file(buffer)
            messages.log('Using logged_hf_token_value')
            self.assertEqual(buffer.getvalue().strip(), 'Using ***')
        finally:
            messages.set_message_file(previous_file)
            if previous is None:
                os.environ.pop('HF_TOKEN', None)
            else:
                os.environ['HF_TOKEN'] = previous

    def test_log_can_skip_censoring(self):
        previous = os.environ.get('HF_TOKEN')
        os.environ['HF_TOKEN'] = 'logged_hf_token_value'
        buffer = io.StringIO()
        previous_file = messages.get_message_file()
        try:
            messages.set_message_file(buffer)
            messages.log('logged_hf_token_value', censor_secrets_output=False)
            self.assertEqual(buffer.getvalue().strip(), 'logged_hf_token_value')
        finally:
            messages.set_message_file(previous_file)
            if previous is None:
                os.environ.pop('HF_TOKEN', None)
            else:
                os.environ['HF_TOKEN'] = previous


if __name__ == '__main__':
    unittest.main()
