import urllib.parse

import requests
import requests.exceptions

import dgenerate.batchprocess.util as _b_util
import dgenerate.messages as _messages
import dgenerate.subcommands.subcommand as _subcommand


class CivitAILinksSubCommand(_subcommand.SubCommand):
    """
    Utility to fetch hard links for every model on a CivitAI page.

    API token is optional, it will simply append it to the generated hard links for you.

    Examples:

    dgenerate --sub-command civitai-links https://civitai.com/models/4384/dreamshaper

    dgenerate --sub-command civitai-links https://civitai.com/models/4384/dreamshaper --token $CIVIT_AI_TOKEN

    See: dgenerate --sub-command civitai-links --help
    """

    NAMES = ['civitai-links']

    @staticmethod
    def _get_model_data(model_id):
        """
        Fetch model data from the CivitAI API.

        :param model_id: ID of the model to fetch data for.
        :return: JSON response from the API.
        :raises HTTPError: If the API request fails.
        """
        url = f"https://civitai.com/api/v1/models/{model_id}"
        try:
            response = requests.get(url)
            response.raise_for_status()  # Check for request errors
            return response.json()
        except requests.exceptions.HTTPError as http_err:
            _messages.error(f"HTTP error occurred: {http_err}")
            raise
        except requests.exceptions.RequestException as req_err:
            _messages.error(f"Error occurred during the request: {req_err}")
            raise

    @staticmethod
    def _extract_links(data, token):
        """
        Extract hard links from model data.

        :param data: Model data in JSON format.
        :param token: Optional API token to append to each link.
        :return: List of formatted hard links.
        """
        hard_links = []
        for version in data.get("modelVersions", []):
            version_name = version.get("name", "Unnamed Version")
            for file in version.get("files", []):
                file_type = file.get("type", "Model")
                base_url = file.get("downloadUrl")
                metadata = file.get("metadata", {})

                # parse the base URL
                url_parts = urllib.parse.urlparse(base_url)
                query_params = urllib.parse.parse_qs(url_parts.query)

                # add metadata and token to query parameters
                for k, v in metadata.items():
                    query_params[k] = [v]
                if token:
                    query_params['token'] = [token]

                # reconstruct the URL with sanitized query parameters
                # the api sometimes returns duplicate parameters
                # which does not result in a usable link
                sanitized_query = urllib.parse.urlencode(query_params, doseq=True)

                sanitized_url = urllib.parse.urlunparse(
                    (url_parts.scheme,
                     url_parts.netloc,
                     url_parts.path,
                     url_parts.params,
                     sanitized_query,
                     url_parts.fragment)
                )

                hard_links.append(f"{version_name} ({file_type}): {sanitized_url}")

        return hard_links

    def __init__(self, program_name='civitai-links', **kwargs):
        super().__init__(**kwargs)

        self._parser = parser = _b_util.DirectiveArgumentParser(
            prog=program_name,
            description='List hard links to models on a CivitAI page.')

        parser.add_argument('url', help='CivitAI model page URL.')
        parser.add_argument('--token', help='Optional API token to auto append to each link.', nargs=1, required=False)

    def __call__(self) -> int:
        """
        Main entry point for the subcommand. Parses arguments, fetches model data,
        extracts links, and logs the results.

        :return: Exit code.
        """

        if self.local_files_only:
            _messages.error("The civitai-links subcommand does not support --offline-mode")
            return 1

        args = self._parser.parse_args(self.args)

        if self._parser.return_code is not None:
            return self._parser.return_code

        try:
            parsed_url = urllib.parse.urlparse(args.url)
        except Exception as e:
            _messages.error(f'URL Parsing syntax error: {e}')
            return 1

        if parsed_url.netloc != 'civitai.com':
            _messages.error(f'Invalid non civitai.com URL given: {args.url}')
            return 1

        try:
            url_parts = parsed_url.path.strip('/').split('/')

            if len(url_parts) < 2:
                _messages.error(
                    f"Failed to process URL: {args.url}, "
                    f"could not extract model id.")
                return 1

            model_id = url_parts[1].strip()
            if not model_id:
                _messages.error(f"Failed to process URL: {args.url}, "
                                f"could not extract model id.")
                return 1

            data = self._get_model_data(model_id)
            links = self._extract_links(data, args.token[0] if args.token else None)

        except (requests.exceptions.HTTPError,
                requests.exceptions.RequestException) as e:
            _messages.error(f"Failed API request for: {args.url}")
            return 1

        _messages.log(f'Models at: {args.url}', underline=True)
        _messages.log()

        for link in links:
            _messages.log(link)

        return 0
