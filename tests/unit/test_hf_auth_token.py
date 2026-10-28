import os
import unittest
import unittest.mock
import urllib.parse

import dgenerate.arguments as arguments
import dgenerate.hfhub as hfhub
import dgenerate.webcache as webcache


class TestResolveAuthToken(unittest.TestCase):

    def test_explicit_token_wins(self):
        with hfhub.auth_token_context('from-context'), \
                unittest.mock.patch('huggingface_hub.get_token', return_value='from-login'):
            self.assertEqual(hfhub.resolve_auth_token('  explicit  '), 'explicit')

    def test_context_beats_login(self):
        with hfhub.auth_token_context('from-context'), \
                unittest.mock.patch('huggingface_hub.get_token', return_value='from-login'):
            self.assertEqual(hfhub.resolve_auth_token(None), 'from-context')
            self.assertEqual(hfhub.resolve_auth_token('   '), 'from-context')

    def test_login_when_nothing_else_is_set(self):
        with unittest.mock.patch('huggingface_hub.get_token', return_value='from-login'):
            self.assertEqual(hfhub.resolve_auth_token(None), 'from-login')


class TestAuthTokenContext(unittest.TestCase):

    def test_sets_and_restores_environment(self):
        previous = os.environ.get('HF_TOKEN')
        os.environ['HF_TOKEN'] = 'saved'
        try:
            with hfhub.auth_token_context('session'):
                self.assertEqual(os.environ['HF_TOKEN'], 'session')
                self.assertEqual(hfhub.resolve_auth_token(None), 'session')
            self.assertEqual(os.environ['HF_TOKEN'], 'saved')
            self.assertIsNone(hfhub._AUTH_TOKEN.get())
        finally:
            if previous is None:
                os.environ.pop('HF_TOKEN', None)
            else:
                os.environ['HF_TOKEN'] = previous

    def test_removes_token_it_introduced(self):
        previous = os.environ.pop('HF_TOKEN', None)
        try:
            with hfhub.auth_token_context('session'):
                self.assertEqual(os.environ['HF_TOKEN'], 'session')
            self.assertNotIn('HF_TOKEN', os.environ)
        finally:
            if previous is not None:
                os.environ['HF_TOKEN'] = previous

    def test_empty_token_is_a_no_op(self):
        previous = os.environ.get('HF_TOKEN')
        os.environ['HF_TOKEN'] = 'saved'
        try:
            with hfhub.auth_token_context(''), hfhub.auth_token_context(None):
                self.assertEqual(os.environ['HF_TOKEN'], 'saved')
                self.assertIsNone(hfhub._AUTH_TOKEN.get())
        finally:
            if previous is None:
                os.environ.pop('HF_TOKEN', None)
            else:
                os.environ['HF_TOKEN'] = previous

    def test_nested_contexts_restore(self):
        previous = os.environ.pop('HF_TOKEN', None)
        try:
            with hfhub.auth_token_context('outer'):
                with hfhub.auth_token_context('inner'):
                    self.assertEqual(hfhub.resolve_auth_token(None), 'inner')
                self.assertEqual(os.environ['HF_TOKEN'], 'outer')
                self.assertEqual(hfhub.resolve_auth_token(None), 'outer')
            self.assertNotIn('HF_TOKEN', os.environ)
        finally:
            if previous is not None:
                os.environ['HF_TOKEN'] = previous


class TestParseAuthToken(unittest.TestCase):

    def test_short_and_long_forms(self):
        self.assertEqual(arguments.parse_auth_token(['-atk', 'short']), ('short', []))
        self.assertEqual(arguments.parse_auth_token(['--auth-token', 'long']), ('long', []))
        self.assertEqual(arguments.parse_auth_token(['--auth-token=equals']), ('equals', []))
        self.assertEqual(
            arguments.parse_auth_token(['--model', 'repo', '-atk', 'short']),
            ('short', ['--model', 'repo']))

    def test_empty_value_is_no_token(self):
        self.assertEqual(arguments.parse_auth_token([]), (None, []))
        self.assertEqual(arguments.parse_auth_token(['--model', 'repo']), (None, ['--model', 'repo']))
        self.assertEqual(arguments.parse_auth_token(['--auth-token', '']), (None, []))
        self.assertEqual(arguments.parse_auth_token(['--auth-token=']), (None, []))
        self.assertEqual(arguments.parse_auth_token(['--auth-token', '   ']), (None, []))
        self.assertEqual(arguments.parse_auth_token(['--auth-token', '  spaced  ']), ('spaced', []))

    def test_missing_value_is_a_usage_error(self):
        with self.assertRaises(arguments.DgenerateUsageError):
            arguments.parse_auth_token(['--auth-token'])
        with self.assertRaises(arguments.DgenerateUsageError):
            arguments.parse_auth_token(['-atk'])


class TestCivitaiDownloadToken(unittest.TestCase):

    def _with_token(self, token):
        previous = os.environ.get('CIVIT_AI_TOKEN')
        if token is None:
            os.environ.pop('CIVIT_AI_TOKEN', None)
        else:
            os.environ['CIVIT_AI_TOKEN'] = token
        return previous

    def _restore_token(self, previous):
        if previous is None:
            os.environ.pop('CIVIT_AI_TOKEN', None)
        else:
            os.environ['CIVIT_AI_TOKEN'] = previous

    def test_token_is_appended_for_civitai_com_and_red(self):
        previous = self._with_token('civit-api-token')
        try:
            for url in (
                    'https://civitai.com/api/download/models/3412871?fileId=3305313',
                    'https://www.civitai.com/api/download/models/3412871?fileId=3305313',
                    'https://civitai.red/api/download/models/3412871?fileId=3305313',
                    'https://www.civitai.red/api/download/models/3412871?fileId=3305313',
            ):
                appended = webcache._append_tokens_to_url(url)
                query = dict(urllib.parse.parse_qsl(urllib.parse.urlparse(appended).query))
                self.assertEqual(query['fileId'], '3305313')
                self.assertEqual(query['token'], 'civit-api-token')
        finally:
            self._restore_token(previous)

    def test_existing_token_is_kept(self):
        previous = self._with_token('from-env')
        try:
            url = 'https://civitai.red/api/download/models/1?token=already-set'
            self.assertEqual(webcache._append_tokens_to_url(url), url)
        finally:
            self._restore_token(previous)

    def test_non_download_and_other_hosts_are_unchanged(self):
        previous = self._with_token('civit-api-token')
        try:
            page = 'https://civitai.red/models/4384/dreamshaper'
            other = 'https://example.com/api/download/models/1'
            self.assertEqual(webcache._append_tokens_to_url(page), page)
            self.assertEqual(webcache._append_tokens_to_url(other), other)
        finally:
            self._restore_token(previous)

    def test_missing_token_leaves_the_url_unchanged(self):
        previous = self._with_token(None)
        try:
            url = 'https://civitai.red/api/download/models/3412871?fileId=3305313'
            self.assertEqual(webcache._append_tokens_to_url(url), url)
        finally:
            self._restore_token(previous)


class TestHuggingFaceAuthorizationHeader(unittest.TestCase):

    def test_only_huggingface_hosts(self):
        header = webcache.huggingface_authorization_header(
            'https://huggingface.co/org/repo/resolve/main/model.safetensors',
            auth_token='hf-token')
        self.assertEqual(header, {'Authorization': 'Bearer hf-token'})
        self.assertEqual(
            webcache.huggingface_authorization_header(
                'https://cdn-lfs.hf.co/org/repo/file', auth_token='hf-token'),
            {'Authorization': 'Bearer hf-token'})

    def test_other_hosts_are_left_alone(self):
        self.assertEqual(
            webcache.huggingface_authorization_header(
                'https://civitai.com/api/download/models/1', auth_token='hf-token'),
            {})
        self.assertEqual(
            webcache.huggingface_authorization_header(
                'https://example.com/file.bin', auth_token='hf-token'),
            {})


if __name__ == '__main__':
    unittest.main()
