import copy
from pathlib import Path

from allennlp.common.util import ensure_list
import pytest

from bella_allen_nlp.dataset_readers.tdlstm import TDLSTMDatasetReader

# Cannot subclass from AllenNlpTestCase as it does not work with  
# pytest.mark.parametrize
class TestTDLSTMDatasetReader():
    @pytest.mark.parametrize("lazy", (True, False))
    @pytest.mark.parametrize("incl_target", (False, True))
    @pytest.mark.parametrize("reverse_right_text", [True])
    def test_read_from_file(self, lazy, incl_target, reverse_right_text):
        reader = TDLSTMDatasetReader(lazy=lazy, incl_target=incl_target,
                                     reverse_right_text=reverse_right_text)
        test_fp = Path(__file__, '..', 'test_data', 'target_reader_data.json')
        instances = ensure_list(reader.read(str(test_fp.resolve())))

        instance1 = {"left_text": ["Though", "you", "will", "undoubtedly", "be", 
                                   "seated", "at", "a", "table", "with", "what", 
                                   "seems", "like", "barely", "enough", "room", 
                                   "(", "no", "matter", "what", "the", "size", 
                                   "of", "your", "party", ")", ",", "the", 
                                   "warm", "atomosphere", "is", "worth", "the", 
                                   "cramped", "quarters-", "you" ,"'ll", 
                                   "have", "fun", "and", "forgot", "about",
                                   "the", "tight"],
                     "right_text": [".", "in", "'re", "you"],
                     "target": ["spot"],
                     "sentiment": 'negative'}
        instance5 = {"left_text": ["I", "really", "recommend", "the", "very", 
                                   "simple"],
                     "right_text": ["."],
                     "target": ["Unda", "(", "Egg", ")", "rolls"],
                     "sentiment": 'positive'}
        test_instances = [instance1, instance5]
        if incl_target:
            for test_instance in test_instances:
                target = test_instance['target']
                left_text = test_instance['left_text']
                left_text += target

                right_text = test_instance['right_text']
                target_temp = copy.deepcopy(target)
                target_temp.reverse()
                right_text += target_temp

                test_instance['left_text'] = left_text
                test_instance['right_text'] = right_text

        assert len(instances) == 10
        fields = instances[0].fields
        assert [t.text for t in fields["left_text"].tokens] == instance1["left_text"]
        assert [t.text for t in fields["right_text"].tokens] == instance1["right_text"]
        assert [t.text for t in fields["target"].tokens] == instance1["target"]
        assert fields["label"].label == instance1["sentiment"]
        fields = instances[4].fields
        assert [t.text for t in fields["left_text"].tokens] == instance5["left_text"]
        assert [t.text for t in fields["right_text"].tokens] == instance5["right_text"]
        assert [t.text for t in fields["target"].tokens] == instance5["target"]
        assert fields["label"].label == instance5["sentiment"]