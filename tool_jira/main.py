import logging
import dotenv
import json
import os

from jira import JIRA
import dtlpy as dl

logger = logging.getLogger('[JIRA]')

file_dir = os.path.dirname(os.path.abspath(__file__))

dotenv.load_dotenv(os.path.join(file_dir, '.env'))

JIRA_API_KEY = os.environ['JIRA_API_KEY']
JIRA_USER_EMAIL = os.environ['JIRA_USER_EMAIL']
JIRA_URL = os.environ['JIRA_URL']

jira_page_size = 100
options = {
    'server': JIRA_URL,
    'verify': True
}
jira = JIRA(options=options, basic_auth=(JIRA_USER_EMAIL, JIRA_API_KEY))


class ToolJira(dl.BaseServiceRunner):

    def get_jira_issue_timeline(self, messages: list) -> list:
        tool = messages[-1]["content"]
        tool_args = tool["arguments"]
        logger.info(f"Input args: {tool_args}")
        ticket_key = tool_args["ticket_key"]
        issue = jira.issue(ticket_key, expand='changelog')
        # Ticket creation details
        timeline = [{
            'author': issue.fields.reporter.displayName,
            'date': issue.fields.created,
            'event': 'Ticket Created',
            'summary': issue.fields.summary,
            'description': issue.fields.description
        }]

        # Changelog details
        for history in issue.changelog.histories:
            for item in history.items:
                print(item.field)
                if item.field in ['assignee', 'summary', 'description', 'status']:
                    event_type = f'{item.field.capitalize()} Changed'
                    change_details = {
                        'author': history.author.displayName,
                        'date': history.created,
                        'event': event_type,
                        'from': item.fromString,
                        'to': item.toString
                    }
                elif item.field == 'Link':
                    change_details = {
                        'author': history.author.displayName,
                        'date': history.created,
                        'event': "Linked Ticket Added",
                        'linked': item.toString,
                    }
                else:
                    continue
                timeline.append(change_details)
        # Comments details
        for comment in issue.fields.comment.comments:
            comment_details = {
                'author': comment.author.displayName,
                'date': comment.created,
                'event': 'Comment Added',
                'comment': comment.body
            }
            timeline.append(comment_details)

        timeline = sorted(timeline, key=lambda x: x['date'])
        messages.append({"role": "function",
                         "name": "get_jira_issue_timeline",
                         "content": json.dumps(timeline, indent=2)})
        return messages
