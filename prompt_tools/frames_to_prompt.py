import io
import logging
import os
import tempfile
from typing import List

import dtlpy as dl
from PIL import Image

logger = logging.getLogger('llm-tools.frames-to-prompt')

DEFAULT_GROUP_SIZE = 4
DEFAULT_PROMPT_DIR = '/prompt_items_dir'
DEFAULT_PROMPT_INSTRUCTION = (
    "Analyze these sequential video frames and provide a detailed, search-friendly description. "
    "Include: (1) Key objects, people, and entities visible; "
    "(2) Actions, movements, and events occurring; "
    "(3) Scene setting, location type, and environment; "
    "(4) Any text, signs, or identifiable information; "
    "(5) Notable changes or transitions between frames. "
    "Be specific and factual — mention colors, positions, counts, and directions where applicable."
)

DEFAULT_STITCHED_PROMPT_INSTRUCTION = (
    "The image below is a single image made by stitching sequential video frames horizontally "
    "(left to right in time order). Analyze it and provide a detailed, search-friendly description. "
    "Include: (1) Key objects, people, and entities visible; "
    "(2) Actions, movements, and events occurring; "
    "(3) Scene setting, location type, and environment; "
    "(4) Any text, signs, or identifiable information; "
    "(5) Notable changes or transitions between the frames. "
    "Be specific and factual — mention colors, positions, counts, and directions where applicable."
)


def parse_frame_index(item_name: str) -> int:
    """Extract the local frame index from the item filename.

    The Smart Frames Splitting node names frames as:
        <sub_video_name>_<frame_idx>.<ext>
    e.g. boat-16_000_042.jpg -> 42
    """
    base = os.path.splitext(item_name)[0]
    trailing = base.rsplit('_', 1)[-1]
    return int(trailing)


class ServiceRunner(dl.BaseServiceRunner):

    def get_cycle_items(self, item: dl.Item) -> List[dl.Item]:
        """
        Gets all items in the current pipeline cycle based on the received item's metadata.
        Uses origin_video_name and time to identify items belonging to the same cycle.

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

        # Filter by pipeline run time to isolate frames from the same execution
        run_time = item.metadata.get('time', None)
        if run_time is not None:
            filters.add(field='metadata.time', values=run_time)

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
            frame_indices = sorted([parse_frame_index(i.name) for i in group])

            fps = group[0].metadata.get('fps', None)
            if fps and fps > 0:
                frame_timestamps = [round(idx / fps, 2) for idx in frame_indices]
                timestamps_str = ', '.join(f'{t}s' for t in frame_timestamps)
                temporal_context = f"at timestamps {timestamps_str} into the video segment"
            else:
                frame_timestamps = None
                temporal_context = f"at frame positions {', '.join(str(i) for i in frame_indices)}"

            frames_str = '_'.join(str(i) for i in frame_indices)
            prompt_name = f'video-frames-prompt-{frames_str}'
            prompt_item = dl.PromptItem(name=prompt_name)

            frame_description = (
                f"These {len(items_ids_list)} images are sequential frames extracted from a video "
                f"{temporal_context}. "
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
            uploaded.metadata['user']['frame_indices'] = frame_indices
            if frame_timestamps is not None:
                uploaded.metadata['user']['frame_timestamps'] = frame_timestamps

            origin_video_name = item.metadata.get('origin_video_name', None)
            if origin_video_name is not None:
                uploaded.metadata['origin_video_name'] = origin_video_name
            run_time = item.metadata.get('time', None)
            if run_time is not None:
                uploaded.metadata['time'] = run_time

            # TODO: if want to reset this, also add :
            # "hyde_model_name": "nim-phi-4-multimodal-instruct" 
            # in the embedding model 
            # uploaded.metadata.setdefault('prompt', {})['is_hyde'] = True
            uploaded.update()

            logger.info(f"Uploaded prompt item '{prompt_name}': {uploaded.id}")
            uploaded_items.append(uploaded)

        logger.info(f"Created {len(uploaded_items)} prompt items in {prompt_dir}")
        return uploaded_items

    def run_stitched(self, item: dl.Item, context: dl.Context) -> List[dl.Item]:
        """
        Groups cycle items into batches, stitches each group horizontally into one image,
        and creates a PromptItem with one text + one image (the stitched image).
        All images in a group must have the same dimensions or the run fails.
        """
        logger.info('Running Frames to Stitched Prompt')

        node_config = context.node.metadata.get('customNodeConfig', {})
        self.group_size = node_config.get('group_size', DEFAULT_GROUP_SIZE)
        self.prompt_dir = node_config.get('prompt_dir', DEFAULT_PROMPT_DIR)
        self.prompt_instruction = node_config.get(
            'prompt_instruction', DEFAULT_STITCHED_PROMPT_INSTRUCTION
        )
        logger.info(f"Group size: {self.group_size}")

        self.dataset = item.dataset
        logger.info(f"Dataset: {self.dataset.name}")

        items = self.get_cycle_items(item)
        if not items:
            raise ValueError("No items found in cycle, cannot create prompt items")

        prompt_dir = self.prompt_dir
        logger.info(f"Prompt items will be uploaded to: {prompt_dir}")

        uploaded_items = []

        for group_start in range(0, len(items), self.group_size):
            group = items[group_start:group_start + self.group_size]
            logger.info(f"Processing group starting at position {group_start} with {len(group)} items")

            frame_indices = sorted([parse_frame_index(i.name) for i in group])
            fps = group[0].metadata.get('fps', None)
            if fps and fps > 0:
                frame_timestamps = [round(idx / fps, 2) for idx in frame_indices]
                timestamps_str = ', '.join(f'{t}s' for t in frame_timestamps)
                temporal_context = f"at timestamps {timestamps_str} into the video segment"
            else:
                frame_timestamps = None
                temporal_context = f"at frame positions {', '.join(str(i) for i in frame_indices)}"

            with tempfile.TemporaryDirectory() as tmpdir:
                pil_images = []
                ref_size = None
                for i, frame_item in enumerate(group):
                    path = os.path.join(tmpdir, frame_item.name)
                    frame_item.download(local_path=path)
                    im = Image.open(path).convert('RGB')
                    w, h = im.size
                    if ref_size is None:
                        ref_size = (w, h)
                    elif (w, h) != ref_size:
                        raise ValueError(
                            f"Frame dimensions differ: expected {ref_size}, got ({w}, {h}) for {frame_item.name}"
                        )
                    pil_images.append(im)

                stitch_w = ref_size[0] * len(pil_images)
                stitch_h = ref_size[1]
                stitched = Image.new('RGB', (stitch_w, stitch_h))
                for i, im in enumerate(pil_images):
                    stitched.paste(im, (i * ref_size[0], 0))

                buf = io.BytesIO()
                stitched.save(buf, format='JPEG')
                buf.seek(0)

                frames_str = '_'.join(str(i) for i in frame_indices)
                stitched_name = f'stitched-{frames_str}.jpg'
                stitched_path = os.path.join(tmpdir, stitched_name)
                with open(stitched_path, 'wb') as f:
                    f.write(buf.getvalue())

                stitched_uploaded = self.dataset.items.upload(
                    local_path=stitched_path, remote_path=prompt_dir
                )

            frame_description = (
                f"This image is {len(group)} sequential video frames stitched horizontally (left to right) "
                f"{temporal_context}. "
                f"{self.prompt_instruction}"
            )

            prompt_name = f'video-frames-stitched-prompt-{frames_str}'
            prompt_item = dl.PromptItem(name=prompt_name)
            content = [
                {'mimetype': dl.PromptType.TEXT, 'value': frame_description},
                {'mimetype': dl.PromptType.IMAGE, 'value': stitched_uploaded.stream},
            ]
            prompt_item.add(message={'role': 'user', 'content': content})

            uploaded = self.dataset.items.upload(prompt_item, remote_path=prompt_dir)

            uploaded.metadata['user'] = uploaded.metadata.get('user', {})
            uploaded.metadata['user']['frame_indices'] = frame_indices
            if frame_timestamps is not None:
                uploaded.metadata['user']['frame_timestamps'] = frame_timestamps
            origin_video_name = item.metadata.get('origin_video_name', None)
            if origin_video_name is not None:
                uploaded.metadata['origin_video_name'] = origin_video_name
            run_time = item.metadata.get('time', None)
            if run_time is not None:
                uploaded.metadata['time'] = run_time
            uploaded.update()

            logger.info(f"Uploaded prompt item '{prompt_name}': {uploaded.id}")
            uploaded_items.append(uploaded)

        logger.info(f"Created {len(uploaded_items)} prompt items in {prompt_dir}")
        return uploaded_items
