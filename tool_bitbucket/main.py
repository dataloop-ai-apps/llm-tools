import logging
import requests
import dotenv
import json
import os
from requests.auth import HTTPBasicAuth
import dtlpy as dl

logger = logging.getLogger('[BITBUCKET]')
file_dir = os.path.dirname(os.path.abspath(__file__))
dotenv.load_dotenv(os.path.join(file_dir, '.env'))

BITBUCKET_USERNAME = os.environ['BITBUCKET_USERNAME']
BITBUCKET_PASSWORD = os.environ['BITBUCKET_PASSWORD']
BITBUCKET_WORKSPACE = os.environ['BITBUCKET_WORKSPACE']

base_url = "https://api.bitbucket.org/2.0"


class ToolJira(dl.BaseServiceRunner):

    def get_diff_from_pr(self, messages: list) -> list:
        tool = messages[-1]["content"]
        tool_args = tool["arguments"]
        logger.info(f"Input args: {tool_args}")
        repo_slug = tool_args["repo_slug"]
        pr_id = tool_args["pr_id"]

        auth = HTTPBasicAuth(BITBUCKET_USERNAME, BITBUCKET_PASSWORD)
        # Define the Bitbucket API endpoint for the pull request
        url = f'{base_url}/repositories/{BITBUCKET_WORKSPACE}/{repo_slug}/pullrequests/{pr_id}'
        # Make the request to fetch the pull request details
        response = requests.get(url, auth=auth)
        print(response.status_code, response.text)
        # Check the response status code
        output = dict()
        if response.status_code == 200:
            pull_request = response.json()
            output.update(dict(title=pull_request['title'],
                               description=pull_request['description'],
                               author=pull_request['author']['display_name'],
                               state=pull_request['state']))
        else:
            raise ValueError(f'failed in response. code: {response.status_code}, error: {response.text}')

        url = f'{base_url}/repositories/{BITBUCKET_WORKSPACE}/{repo_slug}/pullrequests/{pr_id}/diff'
        response = requests.get(url, auth=auth)

        if response.status_code == 200:
            output.update(dict(diff=response.text))
        else:
            raise ValueError(f'failed in response. code: {response.status_code}, error: {response.text}')

        messages.append({"role": "function",
                         "name": "get_jira_issue_timeline",
                         "content": json.dumps(output, indent=2)})
        return messages
