from pathlib import Path

from allennlp.common.checks import ConfigurationError
from allennlp.common.params import Params
from allennlp.models import Model
from allennlp.common.testing import ModelTestCase
from flaky import flaky
import pytest

class TargetClassifierTest(ModelTestCase):
    def setUp(self):
        super().setUp()
        #self.TESTS_ROOT = Path(__file__, '..')
        test_dir = Path(__file__, '..', '..', 'test_data')
        test_data_fp = str(Path(test_dir, 'target_reader_data.json').resolve())
        test_config_fp = str(Path(test_dir, 'target_model_config.json').resolve())

        self.set_up_model(test_config_fp,
                          test_data_fp)

    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)

    @flaky
    def test_batch_predictions_are_consistent(self):
        self.ensure_batch_predictions_are_consistent()

    def test_without_target(self):
        # Test that an error raises if the left text encoder does not have an 
        # input dimension that is equal to the context text word embeddings + 
        # the output dimension of the target encoder.
        params = Params.from_file(self.param_file)
        params["model"].pop("target_encoder")
        params["model"]['classifier_feedforward']['input_dim'] = 2
        test_param_fp = Path(self.TEST_DIR, 'without_target_param_file.json')
        params.to_file(str(test_param_fp))
        self.ensure_model_can_train_save_and_load(test_param_fp)
    
    def test_text_target_feed_dims(self):
        '''
        Tests the case when the target and text encoder output dim is different 
        when summed to that of the input feedforward classifier
        '''
        params = Params.from_file(self.param_file).duplicate()
        params["model"]["target_encoder"]["embedding_dim"] = 3
        with pytest.raises(ConfigurationError):
            Model.from_params(vocab=self.vocab, params=params.get('model'))
        
        params = Params.from_file(self.param_file).duplicate()
        params["model"]["text_encoder"]["embedding_dim"] = 3
        with pytest.raises(ConfigurationError):
            Model.from_params(vocab=self.vocab, params=params.get('model'))
        
        params = Params.from_file(self.param_file).duplicate()
        params["model"]["target_encoder"]["embedding_dim"] = 3
        params["model"]["text_encoder"]["embedding_dim"] = 3
        with pytest.raises(ConfigurationError):
            Model.from_params(vocab=self.vocab, params=params.get('model'))

    def test_target_embedding(self):
        '''
        Ensure that the target embedding is the same as the text embedding when 
        None and different when it is specified to be a different embedding
        '''
        params = Params.from_file(self.param_file).duplicate()
        params['model']['target_encoder']['embedding_dim'] = 4
        params['model']['classifier_feedforward']['input_dim'] = 6
        with pytest.raises(ConfigurationError):
            Model.from_params(vocab=self.vocab, params=params.get('model'))
        
        params = Params.from_file(self.param_file).duplicate()
        params['model']['target_field_embedder']['token_embedders']['tokens']['embedding_dim'] = 4
        with pytest.raises(ConfigurationError):
            Model.from_params(vocab=self.vocab, params=params.get('model'))

