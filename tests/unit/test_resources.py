import io
import json
import unittest
from unittest import mock

import dgenerate.resources as resources


class TestCurrentReleaseInfo(unittest.TestCase):
    def test_network_install_optional_in_json(self):
        payload = {
            'version': '6.0.0',
            'commit': 'abc',
            'branch': 'master',
            'pre_release': False,
        }
        info = resources.CurrentReleaseInfo.json_load(
            io.StringIO(json.dumps(payload)))
        self.assertFalse(info.network_install)

    def test_network_install_true(self):
        payload = {
            'version': '6.0.0',
            'commit': 'abc',
            'branch': 'version_6.0.0',
            'pre_release': True,
            'network_install': True,
        }
        info = resources.CurrentReleaseInfo.json_load(
            io.StringIO(json.dumps(payload)))
        self.assertTrue(info.network_install)


class TestCommitsEqual(unittest.TestCase):
    def test_full_hashes_equal(self):
        sha = 'a' * 40
        self.assertTrue(resources.commits_equal(sha, sha.upper()))

    def test_short_prefix_matches_full(self):
        full = 'abcdef0123456789' * 2 + 'abcd'
        short = full[:12]
        self.assertTrue(resources.commits_equal(full, short))


class TestCompareVersions(unittest.TestCase):
    def test_github_tag_prefix(self):
        result = resources.compare_versions('v5.0.0', '4.0.0')
        self.assertEqual(result, resources.VersionComparison.V1_NEWER)


class TestMasterSupersedingRelease(unittest.TestCase):
    @mock.patch('dgenerate.resources.requests.get')
    def test_points_at_newest_tag_on_master(self, mock_get):
        def respond(url, **kwargs):
            response = mock.Mock()
            response.raise_for_status = mock.Mock()
            if url.endswith('version_6.0.0...master'):
                response.json.return_value = {'status': 'ahead', 'ahead_by': 4, 'behind_by': 0}
            elif url.endswith('/tags?per_page=30'):
                response.json.return_value = [
                    {'name': 'pre-release'},
                    {'name': 'v6.0.0'},
                    {'name': 'v5.0.0'},
                ]
            elif url.endswith('v6.0.0...master'):
                response.json.return_value = {'status': 'identical'}
            else:
                response.json.return_value = {'status': 'diverged'}
            return response

        mock_get.side_effect = respond

        result = resources.check_master_superseding_release('version_6.0.0', '5.0.0')

        self.assertIsNotNone(result)
        self.assertEqual(result.tag_name, 'v6.0.0')
        self.assertTrue(result.release_url.endswith('/releases/tag/v6.0.0'))

    @mock.patch('dgenerate.resources.requests.get')
    def test_same_version_tag_moves_to_release(self, mock_get):
        def respond(url, **kwargs):
            response = mock.Mock()
            response.raise_for_status = mock.Mock()
            if url.endswith('/tags?per_page=30'):
                response.json.return_value = [
                    {'name': 'pre-release'},
                    {'name': 'v6.0.0'},
                    {'name': 'v5.0.0'},
                ]
            elif url.endswith('v6.0.0...master'):
                response.json.return_value = {'status': 'identical'}
            else:
                response.json.return_value = {'status': 'ahead', 'ahead_by': 4, 'behind_by': 0}
            return response

        mock_get.side_effect = respond

        result = resources.check_master_superseding_release('version_6.0.0', '6.0.0')

        self.assertIsNotNone(result)
        self.assertEqual(result.tag_name, 'v6.0.0')

    @mock.patch('dgenerate.resources.requests.get')
    def test_older_tag_on_master_is_not_offered(self, mock_get):
        def respond(url, **kwargs):
            response = mock.Mock()
            response.raise_for_status = mock.Mock()
            if url.endswith('/tags?per_page=30'):
                response.json.return_value = [
                    {'name': 'pre-release'},
                    {'name': 'v5.0.0'},
                ]
            elif url.endswith('v5.0.0...master'):
                response.json.return_value = {'status': 'ahead'}
            else:
                response.json.return_value = {'status': 'ahead', 'ahead_by': 4, 'behind_by': 0}
            return response

        mock_get.side_effect = respond

        self.assertIsNone(
            resources.check_master_superseding_release('version_6.0.0', '6.0.0'))

    @mock.patch('dgenerate.resources.requests.get')
    def test_diverged_branch_is_not_superseded(self, mock_get):
        response = mock.Mock()
        response.raise_for_status = mock.Mock()
        response.json.return_value = {'status': 'diverged', 'ahead_by': 2, 'behind_by': 3}
        mock_get.return_value = response

        self.assertIsNone(resources.check_master_superseding_release('version_6.0.0', '6.0.0'))
        mock_get.assert_called_once()


class TestCheckPreReleaseBranchUpdate(unittest.TestCase):
    @mock.patch('dgenerate.resources.requests.get')
    def test_ahead_by_shows_update(self, mock_get):
        compare_response = mock.Mock()
        compare_response.raise_for_status = mock.Mock()
        compare_response.json.return_value = {
            'ahead_by': 3,
            'commits': [{'sha': 'b' * 40}],
        }
        mock_get.return_value = compare_response

        result = resources.check_pre_release_branch_update('a' * 40, 'version_6.0.0')

        self.assertIsNotNone(result)
        self.assertEqual(result.ahead_by, 3)
        self.assertEqual(result.latest_commit, 'b' * 40)
        mock_get.assert_called_once()

    @mock.patch('dgenerate.resources.requests.get')
    def test_up_to_date(self, mock_get):
        local = 'c' * 40
        compare_response = mock.Mock()
        compare_response.raise_for_status = mock.Mock()
        compare_response.json.return_value = {
            'ahead_by': 0,
            'commits': [],
        }
        mock_get.return_value = compare_response

        result = resources.check_pre_release_branch_update(local, 'version_6.0.0')

        self.assertIsNotNone(result)
        self.assertEqual(result.ahead_by, 0)

    def test_missing_commit_returns_none(self):
        self.assertIsNone(
            resources.check_pre_release_branch_update('', 'version_6.0.0'))


class TestReleaseInstallerDownload(unittest.TestCase):
    @mock.patch('dgenerate.resources.installer_asset_prefix')
    @mock.patch('dgenerate.resources.requests.get')
    def test_fetch_asset_for_platform(self, mock_get, mock_prefix):
        mock_prefix.return_value = 'dgenerate-installer-win-'
        mock_get.return_value = mock.Mock(
            raise_for_status=mock.Mock(),
            json=mock.Mock(return_value={
                'assets': [{
                    'name': 'dgenerate-installer-win-6.0.0.exe',
                    'browser_download_url': 'https://example.com/installer.exe',
                }],
            }),
        )

        result = resources.fetch_release_installer_download('v6.0.0')

        self.assertEqual(
            result,
            ('dgenerate-installer-win-6.0.0.exe',
             'https://example.com/installer.exe'))
        self.assertIn('/releases/tags/v6.0.0', mock_get.call_args.args[0])


class TestNetworkInstallFlag(unittest.TestCase):
    def test_flag_enables_network_install(self):
        release_info = resources.CurrentReleaseInfo(
            '6.0.0', 'abc', 'version_6.0.0', True, network_install=True)
        with mock.patch.object(
                resources, 'get_release_info', return_value=release_info):
            self.assertTrue(resources.is_network_install())

    def test_missing_flag_is_not_a_network_install(self):
        release_info = resources.CurrentReleaseInfo(
            '6.0.0', 'abc', 'version_6.0.0', True, network_install=False)
        with mock.patch.object(
                resources, 'get_release_info', return_value=release_info):
            self.assertFalse(resources.is_network_install())


if __name__ == '__main__':
    unittest.main()
