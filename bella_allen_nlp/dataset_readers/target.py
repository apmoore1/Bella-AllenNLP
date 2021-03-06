import json
import logging
from typing import Callable, Union, Dict, List

from allennlp.data.fields import LabelField, TextField, ArrayField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from bella.contexts import context
import numpy as np
from overrides import overrides


logger = logging.getLogger(__name__)


@DatasetReader.register("target_dataset")
class TargetDatasetReader(DatasetReader):
    def __init__(self, lazy: bool = False, 
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 sentiment_mapper: Dict[int, str] = None):
        '''
        This dataset reader can also be used in conjunction with the augmented 
        iterator.

        :param sentiment_mapper: If not given maps -1, 0, 1 labels to `negative`
                                 , `neutral`, and `positive` respectively.
        '''
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or \
                               {"tokens": SingleIdTokenIndexer()}
        self.sentiment_mapper = sentiment_mapper or \
                                {-1: 'negative', 0: 'neutral', 1: 'positive'}
        
    @overrides
    def _read(self, file_path):
        print(file_path)
        with open(file_path, "r") as data_file:
            # This should be a logger
            logger.info("Reading instances from lines in file at: "
                        f"{file_path}")
            for line in data_file:
                line = line.strip("\n")
                if not line:
                    continue
                target_data = json.loads(line)
                text = target_data['text']
                target = target_data['target']
                epoch_numbers = None
                if 'epoch_number' in target_data:
                    epoch_numbers = list(target_data['epoch_number'])
                sentiment = self.sentiment_mapper[target_data['sentiment']]
                yield self.text_to_instance(text, target, sentiment, 
                                            epoch_numbers)
    
    @overrides
    def text_to_instance(self, text: str, target: str, 
                         sentiment: Union[int, None] = None,
                         epoch_numbers: Union[List[int], None]= None
                         ) -> Instance:
        tokenised_text = self._tokenizer.tokenize(text)
        tokenised_target = self._tokenizer.tokenize(target)
        text_field = TextField(tokenised_text, self._token_indexers)
        target_field = TextField(tokenised_target, self._token_indexers)
        fields = {'text': text_field, 'target': target_field}
        if epoch_numbers is not None:
            fields['epoch_numbers'] = ArrayField(np.array(epoch_numbers))
        if sentiment is not None:
            fields['label'] = LabelField(sentiment, skip_indexing=False)
        return Instance(fields)