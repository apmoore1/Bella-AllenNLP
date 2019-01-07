import copy
from pathlib import Path

import pytest

from allennlp.common.testing import ModelTestCase
from allennlp.common.params import Params
from allennlp.models import Model
from flaky import flaky
from allennlp.common.checks import ConfigurationError


class TDLSTMClassifierTest(ModelTestCase):
    def setUp(self):
        super().setUp()

        self._test_dir = Path(__file__, '..', '..', 'test_data')
        test_data_fp = str(Path(self._test_dir, 
                                'target_reader_data.json').resolve())
        test_config_fp = str(Path(self._test_dir, 
                                  'tdlstm_model_config.json').resolve())

        self.set_up_model(test_config_fp,
                          test_data_fp)

    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)

    @flaky
    def test_batch_predictions_are_consistent(self):
        self.ensure_batch_predictions_are_consistent()

    def test_tclstm_version(self):
        # Test the normal case
        tclstm_param_fp = Path(self._test_dir, 'tclstm_model_config.json').resolve()
        self.ensure_model_can_train_save_and_load(tclstm_param_fp)
        # Test that an error raises if the left text encoder does not have an 
        # input dimension that is equal to the context text word embeddings + 
        # the output dimension of the target encoder.
        params = Params.from_file(tclstm_param_fp)
        params["model"]["left_text_encoder"]["embedding_dim"] = 10
        with pytest.raises(ConfigurationError):
            Model.from_params(vocab=self.vocab, params=params.get('model'))
        # Test the right text encoder
        params = Params.from_file(tclstm_param_fp)
        params["model"]["right_text_encoder"]["embedding_dim"] = 10
        with pytest.raises(ConfigurationError):
            Model.from_params(vocab=self.vocab, params=params.get('model'))
        # Test the target encoder
        params = Params.from_file(tclstm_param_fp)
        params["model"]["target_encoder"]["embedding_dim"] = 5
        with pytest.raises(ConfigurationError):
            Model.from_params(vocab=self.vocab, params=params.get('model'))

    def test_incl_target_tdlstm(self):
        # Test whether it can handle not including the target, the main problem 
        # here is whether it can handle left and/or right contexts not 
        # containing any text which is what happens in the last example for the 
        # left context with regards to the `target_reader_data.json` data
        param_file = self.param_file
        params = Params.from_file(param_file).duplicate()
        params["dataset_reader"]["incl_target"] = False
        test_param_fp = Path(self.TEST_DIR, 'tdlstm_incl_param_file.json')
        params.to_file(str(test_param_fp))
        self.ensure_model_can_train_save_and_load(test_param_fp)
    
    def test_incl_target_tclstm(self):
        # Test whether it can handle not including the target, the main problem 
        # here is whether it can handle left and/or right contexts not 
        # containing any text which is what happens in the last example for the 
        # left context with regards to the `target_reader_data.json` data
        param_file = Path(self._test_dir, 'tclstm_model_config.json').resolve()
        params = Params.from_file(param_file).duplicate()
        params["dataset_reader"]["incl_target"] = False
        test_param_fp = Path(self.TEST_DIR, 'tclstm_incl_param_file.json')
        params.to_file(str(test_param_fp))
        self.ensure_model_can_train_save_and_load(test_param_fp)

    def test_target_field_embedder(self):
        # Test that can handle having a target embedder as well as a text
        # embedder
        param_file = Path(self._test_dir, 'tclstm_model_config.json').resolve()
        params = Params.from_file(param_file).duplicate()
        target_embedder = {"token_embedders": {"tokens": {"type": "embedding",
                                                          "embedding_dim": 15,
                                                          "trainable": False}}}
        params['model']['target_field_embedder'] = target_embedder
        params_copy = copy.deepcopy(params)
        with pytest.raises(ConfigurationError):
            Model.from_params(vocab=self.vocab, params=params_copy.get('model'))
        params['model']['target_encoder']['embedding_dim'] = 15
        params['model']['left_text_encoder']['embedding_dim'] = 25
        params['model']['right_text_encoder']['embedding_dim'] = 25
        test_param_fp = Path(self.TEST_DIR, 'tclstm_target_field_embedder.json')
        params.to_file(str(test_param_fp))
        self.ensure_model_can_train_save_and_load(test_param_fp)
        