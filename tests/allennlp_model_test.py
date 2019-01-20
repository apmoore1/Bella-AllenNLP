import tempfile
from pathlib import Path

from allennlp.common.util import ensure_list
from allennlp.common.params import Params
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from bella.parsers import semeval_14
import pytest
from flaky import flaky

from bella_allen_nlp import AllenNLPModel
from bella_allen_nlp.dataset_readers.target import TargetDatasetReader
from bella_allen_nlp.dataset_readers.tdlstm import TDLSTMDatasetReader

class TestAllenNLPModel():

    
    test_dir = Path(__file__, '..', 'test_data')
    
    train_data_fp = Path(test_dir, 'data', 'target_collection_train_data.xml')
    test_data_fp = Path(test_dir, 'data', 'target_collection_test_data.xml')
    
    TARGET_DATA = semeval_14(test_data_fp, name='test data')
    TARGET_TRAIN_DATA = semeval_14(train_data_fp, name='train data')
    
    test_model_dir = Path(test_dir, 'model_configs')
    MODEL_TARGET_FP = Path(test_model_dir, 'test_target_model_config.json')
    MODEL_TDLSTM_FP = Path(test_model_dir, 'test_tdlstm_model_config.json')

    def test_repr_test(self):
        model = AllenNLPModel('ML', self.MODEL_TARGET_FP)
        model_repr = model.__repr__()
        assert model_repr == 'ML'

    def test_fitted(self):
        model = AllenNLPModel('ML', self.MODEL_TARGET_FP)
        assert not model.fitted

        model.fitted = True
        assert model.fitted

    @pytest.mark.parametrize("test_data", (True, False))
    def test_fit(self, test_data):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_save_dir_fp = Path(temp_dir, 'test save dir')
            model = AllenNLPModel('ML', self.MODEL_TARGET_FP, temp_save_dir_fp)
            assert not model.fitted
            if test_data:
                model.fit(self.TARGET_TRAIN_DATA, self.TARGET_TRAIN_DATA, 
                          self.TARGET_DATA)
            else:
                model.fit(self.TARGET_TRAIN_DATA, self.TARGET_TRAIN_DATA)
            assert model.fitted
            assert temp_save_dir_fp.is_dir()
            token_index = model.model.vocab.get_token_to_index_vocabulary('tokens')
            if test_data:
                assert 'Tais' in list(token_index.keys())
            else:
                assert 'Tais' not in list(token_index.keys())

    @pytest.mark.parametrize("test_data", (True, False))
    def test_load(self, test_data):
        # Testing that an error is raised when there is no save directory
        model = AllenNLPModel('ML', self.MODEL_TARGET_FP)
        with pytest.raises(Exception):
            model.load()
        model.fit(self.TARGET_TRAIN_DATA, self.TARGET_TRAIN_DATA)
        assert model.fitted
        with pytest.raises(Exception):
            model.load()
        # Testing when the save directory is given
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_save_dir_fp = Path(temp_dir, 'test save dir')
            model = AllenNLPModel('ML', self.MODEL_TARGET_FP, temp_save_dir_fp)
            with pytest.raises(FileNotFoundError):
                model.load()
            with pytest.raises(FileNotFoundError):
                temp_save_dir_fp.mkdir()
                model.load()
            temp_save_dir_fp.rmdir()
            if test_data:
                model.fit(self.TARGET_TRAIN_DATA, self.TARGET_TRAIN_DATA, 
                          self.TARGET_DATA)
            else:
                model.fit(self.TARGET_TRAIN_DATA, self.TARGET_TRAIN_DATA)
            archived_model = model.load()
            archived_model = archived_model.model
            token_index = archived_model.vocab.get_token_to_index_vocabulary('tokens')
            assert len(token_index) > 10
            if test_data:
                assert 'Tais' in list(token_index.keys())
            else:
                assert 'Tais' not in list(token_index.keys())

        

    @flaky
    def test_set_random_seeds(self):
        model_params = Params.from_file(self.MODEL_TARGET_FP.resolve())
        seed_keys = ["random_seed", "numpy_seed", "pytorch_seed"]
        for key in seed_keys:
            assert key not in model_params
        AllenNLPModel._set_random_seeds(model_params)
        seed_values = {}
        for key in seed_keys:
            assert key in model_params
            seed_values[key] = model_params[key]
        AllenNLPModel._set_random_seeds(model_params)
        for key in seed_keys:
            assert seed_values[key] != model_params[key]
        

    @pytest.mark.parametrize("lazy", (True, False))
    def test_get_vocab(self, lazy):
        def tokens_labels_exist(dataset_reader, data_paths, test=False):
            true_labels = sorted(['positive', 'negative', 'neutral'])

            vocab = AllenNLPModel._get_vocab(dataset_reader, data_paths)
            tokens = list(vocab.get_token_to_index_vocabulary('tokens').keys())
            labels = list(vocab.get_token_to_index_vocabulary('labels').keys())
            labels = sorted(labels)
            assert true_labels == labels

            assert len(tokens) > 0
            if test:
                assert 'Tais' in tokens
                assert 'Tha' in tokens
            else:
                assert 'Tais' not in tokens
                assert 'Tha' not in tokens 

        with tempfile.TemporaryDirectory() as temp_data_dir:
            train_fp = Path(temp_data_dir, 'train data')
            AllenNLPModel._data_to_json(self.TARGET_TRAIN_DATA, train_fp)
            val_fp = Path(temp_data_dir, 'val data')
            AllenNLPModel._data_to_json(self.TARGET_TRAIN_DATA, val_fp)
            test_fp = Path(temp_data_dir, 'test data')
            AllenNLPModel._data_to_json(self.TARGET_DATA, test_fp)
                
            dataset_reader = TargetDatasetReader(lazy=lazy)
            tokens_labels_exist(dataset_reader, [train_fp, val_fp], test=False)
            tokens_labels_exist(dataset_reader, [train_fp, val_fp, test_fp], 
                                test=True)
            

    def test_preprocess_and_load_param_file(self):
        fields_to_remove = ['train_data_path', 'validation_data_path', 
                            'test_data_path', 'evaluate_on_test']
        
        model_params = AllenNLPModel._preprocess_and_load_param_file(self.MODEL_TARGET_FP)
        for field in fields_to_remove:
            assert field not in model_params

        model_params = Params.from_file(self.MODEL_TARGET_FP.resolve())
        for field in fields_to_remove:
            assert field in model_params

    def test_add_dataset_paths(self):
        model_params = AllenNLPModel._preprocess_and_load_param_file(self.MODEL_TARGET_FP)
        train_path = Path(self.test_data_fp, 'train_data')
        val_path = Path(self.test_data_fp, 'val_data')
        AllenNLPModel._add_dataset_paths(model_params, train_path, val_path)
        assert str(train_path.resolve()) == model_params['train_data_path']
        assert str(val_path.resolve()) == model_params['validation_data_path']
        with pytest.raises(KeyError):
            model_params['test_data_path']
        
        model_params = AllenNLPModel._preprocess_and_load_param_file(self.MODEL_TARGET_FP)
        test_path = Path(self.test_data_fp, 'test_data')
        AllenNLPModel._add_dataset_paths(model_params, train_path, val_path, test_path)
        assert str(train_path.resolve()) == model_params['train_data_path']
        assert str(val_path.resolve()) == model_params['validation_data_path']
        assert str(test_path.resolve()) == model_params['test_data_path']

    @pytest.mark.parametrize("dataset_reader", [TargetDatasetReader(),
                                                TDLSTMDatasetReader()])
    def test_data_to_json(self, dataset_reader: DatasetReader):
        model = AllenNLPModel('ML', self.MODEL_TARGET_FP)
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file = Path(temp_dir, 'temp_file.json')
            model._data_to_json(self.TARGET_DATA, temp_file)
            # Need to now read it with multiple dataset readers as in TDLSTM and target
            reader = dataset_reader
            instances = ensure_list(reader.read((temp_file)))
            text_1 = ['Tha', 'phone', 'came', 'with', 'a', 'very', 'good', 
                      'battery', 'life', ',', 'however', 'the', 'phone', 'was', 
                      'not', 'very', 'good', '.']
            text_2 = ['Tais', 'was', 'an', 'ok', 'camera', 'but', 'the', 
                      'lens', 'could', 'have', 'been', 'better', '.']
            if isinstance(dataset_reader, TargetDatasetReader):
                test_instances = [{"text": text_1, "target": ["battery", "life"],
                                "sentiment": 'positive'}, {"text": text_1, 
                                "target": ["phone"], "sentiment": 'negative'}, 
                                {"text": text_2, "target": ["camera"], 
                                "sentiment": 'neutral'}, {"text": text_2, 
                                "target": ["lens"], "sentiment": 'negative'}]
            elif isinstance(dataset_reader, TDLSTMDatasetReader):
                left_text_1_1 = text_1[:9]
                right_text_1_1 = text_1[7:]
                test_instance_1_1 = {"left_text": left_text_1_1, 
                                     "right_text": right_text_1_1, 
                                     "target": ["battery", "life"],
                                     "sentiment": 'positive'}
                left_text_1_2 = text_1[:13]
                right_text_1_2 = text_1[12:]
                test_instance_1_2 =  {"left_text": left_text_1_2, 
                                     "right_text": right_text_1_2, 
                                     "target": ["phone"], "sentiment": 'negative'}
                left_text_2_1 = text_2[:5]
                right_text_2_1 = text_2[4:]
                test_instance_2_1 =  {"left_text": left_text_2_1, 
                                     "right_text": right_text_2_1, 
                                     "target": ["camera"], "sentiment": 'neutral'}
                left_text_2_2 = text_2[:8]
                right_text_2_2 = text_2[7:]
                test_instance_2_2 =  {"left_text": left_text_2_2, 
                                     "right_text": right_text_2_2, 
                                     "target": ["lens"], "sentiment": 'negative'}
                test_instances = [test_instance_1_1, test_instance_1_2,
                                  test_instance_2_1, test_instance_2_2]
            
            assert len(test_instances) == len(instances)
            text_keys = ['text', 'target', 'left_text', 'right_text']
            for index, instance in enumerate(instances):
                instance = instance.fields
                test_instance = test_instances[index]
                for key, value in test_instance.items():
                    
                    if key in text_keys:
                        value_other = [token.text for token in instance[key].tokens]
                    elif key == 'sentiment':
                        value_other = instance['label'].label

                    if key == 'right_text':
                        value_other.reverse()
                    assert value == value_other