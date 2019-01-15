from pathlib import Path

from allennlp.common.checks import ConfigurationError
from allennlp.common.params import Params
from allennlp.models import Model
from allennlp.common.testing import ModelTestCase
from flaky import flaky
import pytest

class AttentionTargetClassifierTest(ModelTestCase):
    '''
    This has two parameter files associated to it as there are two different 
    types of models that can be constructed with the Attention Target 
    Classifier. The AT model which uses attention over a sentence and ATAE 
    which is the same as AT but the input text encoder takes as input the 
    text word vectors concatenated with the encoded target of which AT only 
    used the text word vectors. Therefore both of these models are tested here.
    '''
    def setUp(self):
        super().setUp()
        test_dir = Path(__file__, '..', '..', 'test_data')
        test_data_fp = str(Path(test_dir, 'target_reader_data.json').resolve())
        atae_config_fp = str(Path(test_dir, 'atae_target_model_config.json').resolve())
        at_config_fp = str(Path(test_dir, 'at_target_model_config.json').resolve())
        self.param_files = [atae_config_fp, at_config_fp]

        self.set_up_model(atae_config_fp,
                          test_data_fp)

    def test_atate_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)

    def test_at_model_can_train_and_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_files[1])

    @flaky
    def test_batch_predictions_are_consistent(self):
        self.ensure_batch_predictions_are_consistent()

    def test_target_embedding(self):
        '''
        Tests that an error occurs if the Target embedding dimension does 
        not match that of the input to the target encoder.
        '''

        for param_file in self.param_files:
            params = Params.from_file(param_file).duplicate()
            params['model']['target_encoder']['embedding_dim'] = 4
            with pytest.raises(ConfigurationError):
                Model.from_params(vocab=self.vocab, params=params.get('model'))
    
    def test_text_embedding_and_target_encoder(self):
        '''
        Tests that an error occurs if the Text embedding that is used for the 
        target encoder is not the same dimension
        '''
        for param_file in self.param_files:
            params = Params.from_file(param_file).duplicate()
            params['model'].pop('target_field_embedder')
            with pytest.raises(ConfigurationError):
                Model.from_params(vocab=self.vocab, params=params.get('model'))
    
    def test_text_embedding_and_text_encoder(self):
        '''
        Tests that an error occurs if the Text embedding has a different 
        dimension to the text encoder.
        '''
        for param_file in self.param_files:
            params = Params.from_file(param_file).duplicate()
            params['model']['text_field_embedder']['token_embedders']['tokens']['embedding_dim'] = 4
            with pytest.raises(ConfigurationError):
                Model.from_params(vocab=self.vocab, params=params.get('model'))

    def test_atae_text_encoding_input_includes_target_encoding_out(self):
        '''
        Tests that the ATAE raises an error when the text encoder input is 
        not equal to the output dimensions of the text embedding and the 
        target encoder. This also tests that this is not the case for the 
        AT model.
        '''
        # This should raise an error for ATAE but not AT
        for index, param_file in enumerate(self.param_files):
            params = Params.from_file(param_file).duplicate()
            text_embed_dim = params['model']['text_field_embedder']['token_embedders']['tokens']['embedding_dim']
            params['model']['text_encoder']['input_size'] = text_embed_dim
            if index == 1:
                Model.from_params(vocab=self.vocab, params=params.get('model'))
                continue
            with pytest.raises(ConfigurationError):
                Model.from_params(vocab=self.vocab, params=params.get('model'))
            

