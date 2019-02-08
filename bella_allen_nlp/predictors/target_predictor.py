from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor
from overrides import overrides

from bella_allen_nlp.dataset_readers.target import TargetDatasetReader
from bella_allen_nlp.dataset_readers.tdlstm import TDLSTMDatasetReader
from bella.contexts import context

@Predictor.register('target-predictor')
class TargetPredictor(Predictor):
    '''
    predict_json return class_probabilities and the labels this is all 
    baseed on decode.
    '''

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like ``{"text": "...", "target": "...."}``.
        Returns that json object as an Instance based on the dataset readers 
        `text_to_instance` method.
        """
        text = json_dict["text"]
        target = json_dict["target"]
        if isinstance(self._dataset_reader, TargetDatasetReader):
            return self._dataset_reader.text_to_instance(text, target)
        # This allows the Target and TDLSTM models to use the same predictors.
        include_target = self._dataset_reader.incl_target
        left_text = context(json_dict, 'left', 
                            inc_target=include_target)[0]
        right_text = context(json_dict, 'right',
                             inc_target=include_target)[0]
        return self._dataset_reader.text_to_instance(left_text, right_text, 
                                                     target)


