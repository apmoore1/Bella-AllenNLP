import json
import logging
from typing import Callable, Union, Dict, List

from allennlp.data.fields import LabelField, TextField, ListField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from bella.contexts import context
from overrides import overrides


logger = logging.getLogger(__name__)


@DatasetReader.register("sentence_target_dataset")
class SentenceTargetDatasetReader(DatasetReader):
    def __init__(self, lazy: bool = False, 
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 sentiment_mapper: Dict[int, str] = None):
        '''
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
                targets = target_data['targets']
                sentiments = [self.sentiment_mapper[sentiment] 
                              for sentiment in target_data['sentiments']]
                yield self.text_to_instance(text, targets, sentiments)
    
    @overrides
    def text_to_instance(self, text: str, targets: List[str], 
                         sentiments: Union[List[str], None] = None
                        ) -> Instance:
        tokenised_text = self._tokenizer.tokenize(text)
        text_field = TextField(tokenised_text, self._token_indexers)
        target_fields = []
        for target in targets:
            tokenised_target = self._tokenizer.tokenize(target)
            target_fields.append(TextField(tokenised_target, 
                                           self._token_indexers))
        target_fields = ListField(target_fields)
        fields = {'text': text_field, 'targets': target_fields}
        if sentiments is not None:
            # Not SequenceLabelField
            sentiment_fields = []
            for sentiment in sentiments:
                sentiment_fields.append(LabelField(sentiment, 
                                        skip_indexing=False))
            fields['labels'] = ListField(sentiment_fields)
        return Instance(fields)