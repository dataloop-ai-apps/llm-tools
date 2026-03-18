import logging

import dtlpy as dl

logger = logging.getLogger('llm-tools.video-to-prompt')

STREAM_URL_TEMPLATE = "https://gate.dataloop.ai/api/v1/items/{item_id}/stream"
DEFAULT_PROMPT_DIR = '/prompt_items_dir'
DEFAULT_PROMPT_INSTRUCTION = (
    "Analyze this video and provide a detailed, search-friendly description. "
    "Include: (1) Key objects, people, and entities visible; "
    "(2) Actions, movements, and events occurring; "
    "(3) Scene setting, location type, and environment; "
    "(4) Any text, signs, or identifiable information; "
    "(5) Notable changes or transitions throughout the video. "
    "Be specific and factual — mention colors, positions, counts, and directions where applicable."
)


class ServiceRunner(dl.BaseServiceRunner):

    def run(self, item: dl.Item, context: dl.Context) -> dl.Item:
        logger.info('Running Video to Prompt')

        node_config = context.node.metadata.get('customNodeConfig', {})
        prompt_dir = node_config.get('prompt_dir', DEFAULT_PROMPT_DIR)
        prompt_instruction = node_config.get('prompt_instruction', DEFAULT_PROMPT_INSTRUCTION)
        model_id = node_config.get('model_id', None)

        video_stream_url = STREAM_URL_TEMPLATE.format(item_id=item.id)

        prompt_item = dl.PromptItem(name=f'video-prompt-{item.id}')

        message = {
            "role": "user",
            "content": [
                {
                    "mimetype": dl.PromptType.TEXT,
                    "value": f"{prompt_instruction} [video_url]({video_stream_url})"
                }
            ]
        }

        model_info = None
        if model_id:
            model = dl.models.get(model_id=model_id)
            model_info = {
                "name": model.name,
                "model_id": model.id,
            }

        prompt_item.add(message=message, model_info=model_info)

        uploaded = item.dataset.items.upload(prompt_item, remote_path=prompt_dir)
        logger.info(f"Uploaded prompt item: {uploaded.id}")

        user_meta_in = item.metadata.get('user', {})
        out_user = uploaded.metadata.setdefault('user', {})
        out_user['origin_video_name'] = user_meta_in.get('origin_video_name', item.name)
        for key in ('time', 'sub_videos_intervals'):
            value = user_meta_in.get(key)
            if value is not None:
                out_user[key] = value
        uploaded = uploaded.update()

        return uploaded
