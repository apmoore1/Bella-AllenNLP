from pathlib import Path

from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor
import pytest

import bella_allen_nlp

class TestTargetPredictor():

    @pytest.mark.parametrize("predictor_type", ('target', 'tdlstm'))
    def test_uses_named_inputs(self, predictor_type):
        inputs = {"spans": [[47, 53]], "target": "prices", "text": "It can "
                  "take a long time for a delivery and the prices are high."}
        model_dir = Path(__file__, '..', '..', 'test_data', 'saved_models')
        model_file = Path(model_dir, 'target_model')
        if predictor_type == 'tdlstm':
            model_file = Path(model_dir, 'tdlstm_model')
        model_file = Path(model_file, 'model.tar.gz').resolve()
        archive = load_archive(model_file)
        predictor = Predictor.from_archive(archive, 'target-predictor')

        prediction_results = predictor.predict_json(inputs)
        class_probabilities = prediction_results.get('class_probabilities')
        label = prediction_results.get('label')
        
        assert class_probabilities is not None
        assert label is not None
        
        assert isinstance(class_probabilities, list)
        assert isinstance(label, str)



        