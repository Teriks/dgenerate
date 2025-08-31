# Copyright (c) 2023, Teriks
#
# dgenerate is distributed under the following BSD 3-Clause License
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in
#    the documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
GitHub API client for downloading dgenerate source code.
"""

import certifi
import json
import os
import ssl
import tempfile
import urllib.error
import urllib.request
import zipfile
from packaging import version as pkg_version
from typing import Dict, List, Optional


class GitHubClient:
    """Client for interacting with GitHub API and downloading repositories."""

    def __init__(self, repo_owner: str = "Teriks", repo_name: str = "dgenerate"):
        self.repo_owner = repo_owner
        self.repo_name = repo_name
        self.api_base = "https://api.github.com"
        self.repo_url = f"{self.api_base}/repos/{repo_owner}/{repo_name}"
        self._ssl_context = self._create_ssl_context()

    def _create_ssl_context(self) -> ssl.SSLContext:
        """Create an SSL context using certifi certificates."""
        context = ssl.create_default_context()
        context.load_verify_locations(certifi.where())
        return context

    def get_latest_release(self) -> Optional[Dict]:
        """Get information about the latest release."""
        try:
            url = f"{self.repo_url}/releases/latest"
            req = urllib.request.Request(url)
            req.add_header('User-Agent', 'dgenerate-network-installer/1.0')
            with urllib.request.urlopen(req, context=self._ssl_context) as response:
                return json.loads(response.read().decode('utf-8'))
        except urllib.error.URLError as e:
            print(f"Error getting latest release: {e}")
            return None

    def get_tags(self, per_page: int = 500) -> List[Dict]:
        """
        Get list of Git tags.

        :param per_page: Number of tags to fetch
        :return: List of tag dictionaries
        """
        try:
            url = f"{self.repo_url}/tags?per_page={per_page}"
            req = urllib.request.Request(url)
            req.add_header('User-Agent', 'dgenerate-network-installer/1.0')
            with urllib.request.urlopen(req, context=self._ssl_context) as response:
                return json.loads(response.read().decode('utf-8'))
        except urllib.error.URLError as e:
            print(f"Error getting tags: {e}")
            return []

    def get_releases(self, per_page: int = 10, include_prereleases: bool = False) -> List[Dict]:
        """
        Get list of releases.

        :param per_page: Number of releases to fetch
        :param include_prereleases: Whether to include pre-releases in the results
        :return: List of release dictionaries, filtered to exclude pre-releases unless include_prereleases is True
        """
        try:
            url = f"{self.repo_url}/releases?per_page={per_page}"
            req = urllib.request.Request(url)
            req.add_header('User-Agent', 'dgenerate-network-installer/1.0')
            with urllib.request.urlopen(req, context=self._ssl_context) as response:
                releases = json.loads(response.read().decode('utf-8'))

                # Filter out pre-releases and the "pre-release" tag unless explicitly requested
                if not include_prereleases:
                    releases = [r for r in releases if
                                not r.get('prerelease', False) and r.get('tag_name') != 'pre-release']

                return releases
        except urllib.error.URLError as e:
            print(f"Error getting releases: {e}")
            return []

    def get_releases_and_tags_combined(self, per_page: int = 500, include_prereleases: bool = False) -> List[Dict]:
        """
        Get combined list of releases and tags, with release descriptions where available.
        Only includes versions 0.18.1 and onward.

        :param per_page: Number of items to fetch
        :param include_prereleases: Whether to include pre-releases in the results
        :return: List of combined release/tag dictionaries
        """
        try:
            # Get both releases and tags
            releases = self.get_releases(per_page=per_page, include_prereleases=include_prereleases)
            tags = self.get_tags(per_page=per_page)

            # Create a map of tag names to release info
            release_map = {r['tag_name']: r for r in releases}

            # Create combined list
            combined = []
            seen_tags = set()

            # Add releases first (they have better metadata)
            for release in releases:
                tag_name = release['tag_name']
                if tag_name != 'pre-release':  # Skip misleading tag
                    # Only include versions 0.18.1 and onward
                    try:
                        version_str = tag_name.lstrip('v')
                        if pkg_version.parse(version_str) >= pkg_version.parse("0.18.1"):
                            combined.append({
                                'tag_name': tag_name,
                                'name': release.get('name', tag_name),
                                'prerelease': release.get('prerelease', False),
                                'published_at': release.get('published_at', ''),
                                'is_release': True
                            })
                            seen_tags.add(tag_name)
                    except:
                        # If version parsing fails, skip it
                        continue

            # Add tags that don't have corresponding releases
            for tag in tags:
                tag_name = tag['name']
                if tag_name not in seen_tags and tag_name != 'pre-release':
                    # Only include versions 0.18.1 and onward
                    try:
                        version_str = tag_name.lstrip('v')
                        if pkg_version.parse(version_str) >= pkg_version.parse("0.18.1"):
                            combined.append({
                                'tag_name': tag_name,
                                'name': tag_name,  # Use tag name as display name
                                'prerelease': False,  # Assume tags are not pre-releases
                                'published_at': '',
                                'is_release': False
                            })
                            seen_tags.add(tag_name)
                    except:
                        # If version parsing fails, skip it
                        continue

            # Sort by semantic version (newest first)
            def version_sort_key(item):
                try:
                    # Handle version strings that might have 'v' prefix
                    tag_name = item['tag_name']
                    if tag_name.startswith('v'):
                        version_str = tag_name[1:]
                    else:
                        version_str = tag_name
                    return pkg_version.parse(version_str)
                except Exception:
                    # If version parsing fails, fall back to string sorting
                    # Put unparseable versions at the end
                    return pkg_version.parse("0.0.0")

            combined.sort(key=version_sort_key, reverse=True)

            return combined

        except Exception as e:
            print(f"Error getting combined releases and tags: {e}")
            return []

    def get_branches(self) -> List[Dict]:
        """Get list of branches."""
        try:
            url = f"{self.repo_url}/branches"
            req = urllib.request.Request(url)
            req.add_header('User-Agent', 'dgenerate-network-installer/1.0')
            with urllib.request.urlopen(req, context=self._ssl_context) as response:
                return json.loads(response.read().decode('utf-8'))
        except urllib.error.URLError as e:
            print(f"Error getting branches: {e}")
            return []

    def download_source_archive(self, ref: str = "master", extract_to: Optional[str] = None,
                                progress_callback=None) -> Optional[str]:
        """
        Download source code archive from GitHub.
        
        Args:
            ref: Branch, tag, or commit SHA to download
            extract_to: Directory to extract to. If None, creates a temporary directory.
            progress_callback: Optional callback function(downloaded_bytes, total_bytes)
            
        Returns:
            Path to extracted directory, or None if failed
        """
        try:
            # Download ZIP archive
            archive_url = f"https://github.com/{self.repo_owner}/{self.repo_name}/archive/{ref}.zip"

            if extract_to is None:
                extract_to = tempfile.mkdtemp(prefix=f"{self.repo_name}_")

            # Create the directory if it doesn't exist
            os.makedirs(extract_to, exist_ok=True)

            # Download and extract
            print(f"Downloading {archive_url}...")

            # Create a request with proper headers and SSL context
            req = urllib.request.Request(archive_url)
            req.add_header('User-Agent', 'dgenerate-network-installer/1.0')

            with urllib.request.urlopen(req, context=self._ssl_context) as response:
                zip_path = os.path.join(extract_to, f"{self.repo_name}-{ref}.zip")

                # Get total file size from headers
                total_size = response.headers.get('content-length')
                if total_size:
                    total_size = int(total_size)

                downloaded_size = 0
                chunk_size = 8192  # 8KB chunks

                with open(zip_path, 'wb') as f:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded_size += len(chunk)

                        # Report progress if callback provided
                        if progress_callback and total_size:
                            progress_callback(downloaded_size, total_size)

                # Extract the ZIP file
                print(f"Extracting to {extract_to}...")
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_to)

                # Remove the ZIP file
                os.remove(zip_path)

                # Find the extracted directory (it will be named repo-branch)
                extracted_dir = None
                for item in os.listdir(extract_to):
                    item_path = os.path.join(extract_to, item)
                    if os.path.isdir(item_path) and item.startswith(f"{self.repo_name}-"):
                        extracted_dir = item_path
                        break

                if extracted_dir:
                    return extracted_dir
                else:
                    print("Could not find extracted directory")
                    return None

        except urllib.error.URLError as e:
            print(f"Error downloading source archive: {e}")
            return None
        except Exception as e:
            print(f"Error extracting source archive: {e}")
            return None

    def get_commits(self, branch: str = "master", per_page: int = 10) -> List[Dict]:
        """Get list of commits from a branch."""
        try:
            url = f"{self.repo_url}/commits?sha={branch}&per_page={per_page}"
            req = urllib.request.Request(url)
            req.add_header('User-Agent', 'dgenerate-network-installer/1.0')
            with urllib.request.urlopen(req, context=self._ssl_context) as response:
                return json.loads(response.read().decode('utf-8'))
        except urllib.error.URLError as e:
            print(f"Error getting commits: {e}")
            return []

    def get_commit_info(self, ref: str = "master") -> Optional[Dict]:
        """Get information about a specific commit."""
        try:
            url = f"{self.repo_url}/commits/{ref}"
            req = urllib.request.Request(url)
            req.add_header('User-Agent', 'dgenerate-network-installer/1.0')
            with urllib.request.urlopen(req, context=self._ssl_context) as response:
                return json.loads(response.read().decode('utf-8'))
        except urllib.error.URLError as e:
            print(f"Error getting commit info: {e}")
            return None

    def search_releases(self, query: str) -> List[Dict]:
        """Search releases by tag name or title."""
        releases = self.get_releases(per_page=100)
        query_lower = query.lower()

        matching_releases = []
        for release in releases:
            if (query_lower in release.get('tag_name', '').lower() or
                    query_lower in release.get('name', '').lower()):
                matching_releases.append(release)

        return matching_releases
