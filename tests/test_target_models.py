from pathlib import Path

from allennlp.common.testing import ModelTestCase

from bella_allen_nlp.allen_models.basic_target_lstm import TargetLSTMClassifier


class TargetLSTMClassifierTest(ModelTestCase):
    def setUp(self):
        super().setUp()
        self.TESTS_ROOT = Path(__file__, '..')
        print('########')
        print(f'{self.TESTS_ROOT.resolve()}')
        print('#####')
        test_dir = Path(__file__, '..', 'test_data')
        test_data_fp = str(Path(test_dir, 'target_reader_data.json').resolve())
        test_config_fp = str(Path(test_dir, 'target_model_config.json').resolve())
        print(test_config_fp)
        print(test_data_fp)

        self.set_up_model(test_config_fp,
                          test_data_fp)

    def test_model_can_train_save_and_load(self):
        print('----------------')
        print(self.param_file)
        print('=-----------')
        self.ensure_model_can_train_save_and_load(self.param_file)