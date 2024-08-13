import logging
import openai
import dotenv
import os
import dtlpy as dl

logger = logging.getLogger('[CHAT]')

file_dir = os.path.dirname(os.path.abspath(__file__))

dotenv.load_dotenv(os.path.join(file_dir, '.env'))
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']

tools = {
    "jira": {
        "description": "Get full history from a jira ticket",
        "parameters": {
            "type": "object",
            "properties": {
                "ticket_key": {"type": "string", "description": "a jira ticket key, e.g. DAT-12749, APPS-736 "}
            },
            "required": ["ticket_key"]
        }
    },
    "bitbucket": {
        "description": "Get a git diff for a PR",
        "parameters": {
            "type": "object",
            "properties": {
                "repo_slug": {"type": "string", "description": "a git repo name, e.g. dtlpy, rubiks"},
                "pr_id": {"type": "number", "description": "an id for the pr, e.g. 1113"}
            },
            "required": ["repo_slug", "pr_id"]
        }
    }
}


class ToolJira(dl.BaseServiceRunner):

    def chat(self, messages, progress: dl.Progress):
        if len(messages) > 10:
            raise ValueError('Cant go over 10 messages')
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            messages=messages,
            model='gpt-4o',
            functions=[{"name": name, **spec} for name, spec in tools.items()],
            function_call="auto"
        )
        ai_message = response.choices[0].message

        if ai_message.function_call:
            tool_name = ai_message.function_call.name
            logger.info(f'Execution tool: {tool_name}')
            progress.update(action=tool_name)
            messages.append({"role": "assistant",
                             "content": None,
                             "function_call": ai_message.function_call}
                            )
        else:
            answer = ai_message.content
            logger.info(f'Returning an answer')
            messages.append({"role": "assistant", "content": answer})
        return messages
