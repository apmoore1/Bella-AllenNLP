import json
import logging
from typing import Callable, Union, Dict, List

from allennlp.data.fields import LabelField, TextField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Tokenizer, WordTokenizer, Token
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from bella.contexts import context
from overrides import overrides


logger = logging.getLogger(__name__)


@DatasetReader.register("tdlstm_dataset")
class TDLSTMDatasetReader(DatasetReader):
    def __init__(self, lazy: bool = False, 
                 tokenizer: Tokenizer = None,
                 incl_target: bool = True,
                 reverse_right_text: bool = True,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 sentiment_mapper: Dict[int, str] = None):
        '''
        :param incl_target: Whether to include the target word(s) in the left 
                            and right contexts. By default this is True as 
                            this is what the original TDLSTM method specified.
        :param reverse_right_text: If the text that can include the target and 
                                   all text right of the target should be 
                                   returned tokenised in reverse order starting 
                                   from the right most token to the left 
                                   most token which would be the first token 
                                   of the target if the target is included. 
                                   This is required to reproduce the single 
                                   layer LSTM method of TDLSTM, if a 
                                   bi-directional LSTM encoder is chosen to 
                                   encode the right text then this parameter 
                                   does not matter and would be quicker to 
                                   choose False.
        :param sentiment_mapper: If not given maps -1, 0, 1 labels to `negative`
                                 , `neutral`, and `positive` respectively.
        '''
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self.incl_target = incl_target
        self.reverse_right_text = reverse_right_text
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
                target = target_data['target']
                left_text = context(target_data, 'left', 
                                    inc_target=self.incl_target)[0]
                right_text = context(target_data, 'right',
                                     inc_target=self.incl_target)[0]
                sentiment = self.sentiment_mapper[target_data['sentiment']]
                yield self.text_to_instance(left_text, right_text, target, 
                                            sentiment)
    
    @overrides
    def text_to_instance(self, left_text: str, right_text: str, target: str, 
                         sentiment: Union[int, None] = None
                        ) -> Instance:
        left_tokenised_text = self._tokenizer.tokenize(left_text)
        right_tokenised_text = self._tokenizer.tokenize(right_text)
        if self.reverse_right_text:
            right_tokenised_text.reverse()
        tokenised_target = self._tokenizer.tokenize(target)
        # If one of the right or left contexts are empty then to avoid having 
        # an empty tensor dict when converting the instances from 
        # batches/datasets add the padding token which is 0
        if left_text.strip() == '':
            left_tokenised_text: List[Token] = [Token(text_id=0)]
        if right_text.strip() == '':
            right_tokenised_text: List[Token] = [Token(text_id=0)]
        left_text_field = TextField(left_tokenised_text, self._token_indexers)
        right_text_field = TextField(right_tokenised_text, self._token_indexers)
        target_field = TextField(tokenised_target, self._token_indexers)

        fields = {'left_text': left_text_field, 'right_text': right_text_field, 
                  'target': target_field}
        if sentiment is not None:
            fields['label'] = LabelField(sentiment, skip_indexing=False)
        return Instance(fields)