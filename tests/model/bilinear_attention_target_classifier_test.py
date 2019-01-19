from pathlib import Path

from allennlp.common.checks import ConfigurationError
from allennlp.common.params import Params
from allennlp.models import Model
from allennlp.common.testing import ModelTestCase
from flaky import flaky
import pytest

class BiLinearAttentionTargetClassifierTest(ModelTestCase):
    def setUp(self):
        super().setUp()
        test_dir = Path(__file__, '..', '..', 'test_data')
        test_data_fp = str(Path(test_dir, 'data', 'target_reader_data.json').resolve())
        test_config_fp = str(Path(test_dir, 'model_configs', 'bilinear_attention_target_model_config.json').resolve())

        self.set_up_model(test_config_fp,
                          test_data_fp)

    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)

    @flaky
    def test_batch_predictions_are_consistent(self):
        self.ensure_batch_predictions_are_consistent()

    def test_target_embedding(self):
        '''
        Tests that an error occurs if the Target embedding dimension does 
        not match that of the input to the target encoder.
        '''
        params = Params.from_file(self.param_file).duplicate()
        params['model']['target_encoder']['embedding_dim'] = 4
        with pytest.raises(ConfigurationError):
            Model.from_params(vocab=self.vocab, params=params.get('model'))
        params = Params.from_file(self.param_file).duplicate()
        params['model']['target_encoder']['embedding_dim'] = 12
        params['model']['target_field_embedder']['token_embedders']['tokens']['embedding_dim'] = 12
        test_param_fp = Path(self.TEST_DIR, 'different_target_text_embedding.json')
        params.to_file(str(test_param_fp))
        self.ensure_model_can_train_save_and_load(test_param_fp)
    
    def test_text_embedding_and_target_encoder(self):
        '''
        Tests that an error occurs if the Text embedding that is used for the 
        target encoder is not the same dimension
        '''
        params = Params.from_file(self.param_file).duplicate()
        params['model'].pop('target_field_embedder')
        params['model']['target_encoder']['embedding_dim'] = 12
        with pytest.raises(ConfigurationError):
            Model.from_params(vocab=self.vocab, params=params.get('model'))
        params = Params.from_file(self.param_file).duplicate()
        params['model'].pop('target_field_embedder')
        test_param_fp = Path(self.TEST_DIR, 'no_target_embedding.json')
        params.to_file(str(test_param_fp))
        self.ensure_model_can_train_save_and_load(test_param_fp)
    
    def test_text_embedding_and_text_encoder(self):
        '''
        Tests that an error occurs if the Text embedding has a different 
        dimension to the text encoder.
        '''
        params = Params.from_file(self.param_file).duplicate()
        params['model']['text_field_embedder']['token_embedders']['tokens']['embedding_dim'] = 4
        with pytest.raises(ConfigurationError):
            Model.from_params(vocab=self.vocab, params=params.get('model'))

