from pathlib import Path

from allennlp.common.testing import ModelTestCase

from bella.allen_models.basic_target_lstm import TargetLSTMClassifier
from bella.dataset_readers.target_reader import TargetDatasetReader


class TargetLSTMClassifierTest(ModelTestCase):
    def setUp(self):
        super().setUp()
        test_dir = Path(__file__, '..', 'test_data')
        test_data_fp = str(Path(test_dir, 'target_reader_data.json').resolve())
        test_config_fp = str(Path(test_dir, 'target_model_config.json').resolve())

        self.set_up_model(test_config_fp,
                          test_data_fp)

    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)