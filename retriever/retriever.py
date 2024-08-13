import dtlpy as dl
import logging
import json

logger = logging.getLogger('[RETRIEVER]')


class Retriever(dl.BaseServiceRunner):

    @staticmethod
    def query_nearest_items_prompt(item: dl.Item,
                                   dataset_id: str,
                                   feature_set_id: str,
                                   embeddings: list,
                                   query: dict = None,
                                   k=30):
        """
        Return k nearest items (from the input dataset) to the input embeddings vector.
        Euclidean distance over features from the input feature_set_id

        :param item: input prompt item, used for saving the nearest items
        :param dataset_id: query over items from this dataset
        :param feature_set_id: query over feature vectors from this feature set
        :param embeddings: the input vector to find similarities
        :param query: input items query to filter the searchable items
        :param k: number of neighbours to return
        :return:
        """
        if item.metadata.get('system', dict()).get('shebang', dict()).get('dltype') != 'prompt':
            raise ValueError(f'Only prompt items are supported. Cant run for item: {item.id}')
        logger.info(f'Starting run with: '
                    f'dataset: {dataset_id}, '
                    f'feature set: {feature_set_id}, '
                    f'item id: {item.id}, '
                    f'query: {query}, '
                    f'k: {k}')

        if isinstance(embeddings, str):
            embeddings = json.loads(embeddings)
        if query is None:
            query = {'$and': [{'hidden': False},
                              {'type': 'file'},
                              {'datasetId': dataset_id}]}
        else:
            logger.info(f'Got input query: {query}')
            if '$and' in query:
                query['$and'].extend([{'hidden': False},
                                      {'type': 'file'},
                                      {'datasetId': dataset_id}])
            else:
                query['$and'] = [{'hidden': False},
                                 {'type': 'file'},
                                 {'datasetId': dataset_id}]
        logger.info(f'Running items query with the following: {query}')

        custom_filter = {
            'filter': query,
            'resource': 'items',
            'join': {
                'on': {
                    'resource': 'feature_vectors',
                    'local': 'entityId',
                    'forigen': 'id'
                },
                'filter': {
                    'value': {
                        '$euclid': {
                            'input': embeddings,
                            '$euclidSort': {'eu_dist': 'ascending'}
                        }
                    },
                    'featureSetId': feature_set_id,
                    'datasetId': dataset_id
                },
            }
        }
        filters = dl.Filters(custom_filter=custom_filter,
                             resource=dl.FiltersResource.ITEM,
                             page_size=k)
        res = dl.datasets.get(dataset_id=dataset_id).items.list(filters=filters)
        nearest_items = res.items[:k]
        # add nearest items to the prompt
        prompt_item = dl.PromptItem.from_item(item)
        prompt_item.prompts[-1].add_element(mimetype=dl.PromptType.METADATA,
                                            value={'nearestItems': [item.id for item in nearest_items]})
        prompt_item.update()
        return item


if __name__ == "__main__":
    self = Retriever()
    ex = dl.executions.get('66ae8fc08c886e28ab62eed7')
    inputs = ex.input
    inputs['item'] = dl.items.get(item_id=inputs['item']['item_id'])
    self.query_nearest_items_prompt(**ex.input)
