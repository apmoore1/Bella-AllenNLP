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
        #self.TESTS_ROOT = Path(__file__, '..')
        self._test_dir = Path(__file__, '..', 'test_data')
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
        