import logging
import os
from typing import List

import dtlpy as dl

logger = logging.getLogger('llm-tools.frames-to-prompt')

DEFAULT_GROUP_SIZE = 4
DEFAULT_PROMPT_DIR = '/prompt_items_dir'
DEFAULT_PROMPT_INSTRUCTION = (
    "Provide a detailed description of this video segment: "
    "describe the scene, setting, and environment; "
    "describe the actions, movements, and interactions taking place; "
    "note any changes or progression between frames; "
)
FRAMES_PER_CHUNK = 250


def frame_index_from_name(item_name: str) -> int:
    """
    Derive the original frame index from the item name.
    Assumes the video was split into chunks of FRAMES_PER_CHUNK frames,
    and the item name follows the pattern: <video>_<chunk>_<frame>.<ext>
    e.g. dancetrack0066_0002_058 -> 0002 * 250 + 58 = 558
    """
    base = os.path.splitext(item_name)[0]
    parts = base.rsplit('_', 2)
    if len(parts) < 3:
        raise ValueError(f"Cannot parse frame index from item name: {item_name}")
    chunk_index = int(parts[-2])
    frame_in_chunk = int(parts[-1])
    return chunk_index * FRAMES_PER_CHUNK + frame_in_chunk


class ServiceRunner(dl.BaseServiceRunner):
    def __init__(self):
        self.group_size = DEFAULT_GROUP_SIZE
        self.prompt_dir = DEFAULT_PROMPT_DIR
        self.prompt_instruction = DEFAULT_PROMPT_INSTRUCTION
        self.dataset = None

    def get_cycle_items(self, item: dl.Item) -> List[dl.Item]:
        """
        Gets all items in the current pipeline cycle based on the received item's metadata.
        Uses origin_video_name and created_time to identify items belonging to the same cycle.

        Args:
            item (dl.Item): Reference item from the wait node

        Returns:
            List[dl.Item]: Sorted list of frame items in the cycle
        """
        input_dir = os.path.dirname(item.filename)
        filters = dl.Filters(field='dir', values=input_dir)

        # Filter by origin video name if available
        original_video_name = item.metadata.get('origin_video_name', None)
        if original_video_name is not None:
            filters.add(field='metadata.origin_video_name', values=original_video_name)

        # Filter by created_time if available
        created_time = item.metadata.get('created_time', None)
        if created_time is not None:
            filters.add(field='metadata.created_time', values=created_time)

        # Filter by sub-video prefix (e.g. MOT16-05-raw_000_122.jpg -> MOT16-05-raw_000_*)
        base = os.path.splitext(item.name)[0]
        sub_video_prefix = base.rsplit('_', 1)[0]
        filters.add(field='name', values=f'{sub_video_prefix}_*')

        items = self.dataset.items.get_all_items(filters=filters)
        logger.info(f"Found {len(items)} items in cycle")

        if not items or len(items) == 0:
            logger.error("No items found in cycle")
            return []

        return sorted(items, key=lambda x: x.name)

    def run(self, item: dl.Item, context: dl.Context) -> List[dl.Item]:
        """
        Groups cycle items into batches and creates a PromptItem for each group.

        Pipeline flow: split video -> smart sampling -> wait node -> this code node.
        This node receives one item from the wait node, retrieves all cycle items,
        groups them, and creates prompt items with text + image elements.

        Args:
            item (dl.Item): Item received from the wait node
            context (dl.Context): Pipeline context containing node configuration

        Returns:
            List[dl.Item]: List of uploaded prompt items
        """
        logger.info('Running Frames to Prompt')

        node_config = context.node.metadata.get('customNodeConfig', {})
        self.group_size = node_config.get('group_size', DEFAULT_GROUP_SIZE)
        self.prompt_dir = node_config.get('prompt_dir', DEFAULT_PROMPT_DIR)
        self.prompt_instruction = node_config.get('prompt_instruction', DEFAULT_PROMPT_INSTRUCTION)
        logger.info(f"Group size: {self.group_size}")

        self.dataset = item.dataset
        logger.info(f"Dataset: {self.dataset.name}")

        # Get all items in the cycle
        items = self.get_cycle_items(item)
        if not items:
            raise ValueError("No items found in cycle, cannot create prompt items")

        prompt_dir = self.prompt_dir
        logger.info(f"Prompt items will be uploaded to: {prompt_dir}")

        uploaded_items = []

        for group_start in range(0, len(items), self.group_size):
            group = items[group_start:group_start + self.group_size]
            logger.info(f"Processing group starting at position {group_start} with {len(group)} items")

            items_ids_list = [i.id for i in group]
            items_frame_index = sorted([frame_index_from_name(i.name) for i in group])

            frames_str = '_'.join(str(i) for i in items_frame_index)
            prompt_name = f'video-frames-prompt-{frames_str}'
            prompt_item = dl.PromptItem(name=prompt_name)

            frame_description = (
                f"These {len(items_ids_list)} images are sequential frames extracted from a video "
                f"at frame indices {', '.join(str(i) for i in items_frame_index)}. "
                f"{self.prompt_instruction}"
            )

            content = [{'mimetype': dl.PromptType.TEXT, 'value': frame_description}]
            for item_id in items_ids_list:
                frame_item = dl.items.get(item_id=item_id)
                content.append({'mimetype': dl.PromptType.IMAGE, 'value': frame_item.stream})

            prompt_item.add(
                message={'role': 'user', 'content': content}
            )

            uploaded = self.dataset.items.upload(prompt_item, remote_path=prompt_dir)

            uploaded.metadata['user'] = uploaded.metadata.get('user', {})
            uploaded.metadata['user']['frame_indices'] = items_frame_index

            origin_video_name = item.metadata.get('origin_video_name', None)
            if origin_video_name is not None:
                uploaded.metadata['origin_video_name'] = origin_video_name
            created_time = item.metadata.get('created_time', None)
            if created_time is not None:
                uploaded.metadata['created_time'] = created_time

            # TODO: if want to reset this, also add :
            # "hyde_model_name": "nim-phi-4-multimodal-instruct" 
            # in the embedding model 
            # uploaded.metadata.setdefault('prompt', {})['is_hyde'] = True
            uploaded.update()

            logger.info(f"Uploaded prompt item '{prompt_name}': {uploaded.id}")
            uploaded_items.append(uploaded)

        logger.info(f"Created {len(uploaded_items)} prompt items in {prompt_dir}")
        return uploaded_items
