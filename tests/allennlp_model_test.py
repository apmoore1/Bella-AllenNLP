import tempfile
from pathlib import Path

from allennlp.common.util import ensure_list
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from bella.parsers import semeval_14
import pytest

from bella_allen_nlp import AllenNLPModel
from bella_allen_nlp.dataset_readers.target import TargetDatasetReader
from bella_allen_nlp.dataset_readers.tdlstm import TDLSTMDatasetReader

class TestAllenNLPModel():

    
    test_dir = Path(__file__, '..', 'test_data')
    test_data_fp = Path(test_dir, 'target_collection_test_data.xml')
    TARGET_DATA = semeval_14(test_data_fp, name='test data')

    def test_repr_test(self):
        model = AllenNLPModel('ML')
        model_repr = model.__repr__()
        assert model_repr == 'ML'

    def test_fitted(self):
        model = AllenNLPModel('ML')
        assert not model.fitted

        model.fitted = True
        assert model.fitted

    @pytest.mark.parametrize("dataset_reader", [TargetDatasetReader(),
                                                TDLSTMDatasetReader()])
    def test_data_to_json(self, dataset_reader: DatasetReader):
        model = AllenNLPModel('ML')
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file = Path(temp_dir, 'temp_file.json')
            model._data_to_json(self.TARGET_DATA, temp_file)
            # Need to now read it with multiple dataset readers as in TDLSTM and target
            reader = dataset_reader
            instances = ensure_list(reader.read((temp_file)))
            text_1 = ['The', 'phone', 'came', 'with', 'a', 'very', 'good', 
                      'battery', 'life', ',', 'however', 'the', 'phone', 'was', 
                      'not', 'very', 'good', '.']
            text_2 = ['This', 'was', 'an', 'ok', 'camera', 'but', 'the', 
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