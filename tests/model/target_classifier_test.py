from pathlib import Path

from allennlp.common.testing import ModelTestCase
from flaky import flaky


class TargetLSTMClassifierTest(ModelTestCase):
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