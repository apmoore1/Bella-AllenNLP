import copy
from pathlib import Path

from allennlp.common.util import ensure_list
import pytest

from bella_allen_nlp.dataset_readers.tdlstm import TDLSTMDatasetReader

# Cannot subclass from AllenNlpTestCase as it does not work with  
# pytest.mark.parametrize
class TestTDLSTMDatasetReader():
    @pytest.mark.parametrize("augmented_data", (True, False))
    @pytest.mark.parametrize("lazy", (True, False))
    @pytest.mark.parametrize("incl_target", (False, True))
    @pytest.mark.parametrize("reverse_right_text", [True, False])
    @pytest.mark.parametrize("sentiment_mapper", [None, {-1: 'Neg', 0: 'Neu', 
                                                         1: 'Pos'}])
    def test_read_from_file(self, lazy, incl_target, reverse_right_text, 
                            sentiment_mapper, augmented_data):
        reader = TDLSTMDatasetReader(lazy=lazy, incl_target=incl_target,
                                     reverse_right_text=reverse_right_text,
                                     sentiment_mapper=sentiment_mapper)
        test_fp = Path(__file__, '..', '..', 'test_data', 'data',
                       'target_reader_data.json')
        if augmented_data:
            test_fp = Path(__file__, '..', '..', 'test_data', 'data', 
                           'augmented_target_reader_data.json')
        instances = ensure_list(reader.read(str(test_fp.resolve())))
        if sentiment_mapper is None:
            sentiment_mapper = {-1: 'negative', 0: 'neutral', 1: 'positive'}

        instance1 = {"left_text": ["Though", "you", "will", "undoubtedly", "be", 
                                   "seated", "at", "a", "table", "with", "what", 
                                   "seems", "like", "barely", "enough", "room", 
                                   "(", "no", "matter", "what", "the", "size", 
                                   "of", "your", "party", ")", ",", "the", 
                                   "warm", "atomosphere", "is", "worth", "the", 
                                   "cramped", "quarters-", "you" ,"'ll", 
                                   "have", "fun", "and", "forgot", "about",
                                   "the", "tight"],
                     "right_text": ["you", "'re", "in", "."],
                     "target": ["spot"],
                     "epoch_number": [-1],
                     "sentiment": sentiment_mapper[-1]}
        instance5 = {"left_text": ["I", "really", "recommend", "the", "very", 
                                   "simple"],
                     "right_text": ["."],
                     "epoch_number": [10, 11, 12],
                     "target": ["Unda", "(", "Egg", ")", "rolls"],
                     "sentiment": sentiment_mapper[1]}
        # Left text in this case is empty but an empty string is represented as 
        # the @@EMPTY_SENTENCE@@ token. However when the target is included 
        # this token should not exist
        instance12 = {"left_text": ["@@EMPTY_SENTENCE@@"],
                      "target": ["lava", "cake", "dessert"], 
                      "right_text": ["was", "incredible", "and", "I", 
                                     "recommend", "it", "."],
                      "epoch_number": [-1],
                      "sentiment": sentiment_mapper[1]}
        # Same as the left context but for the right.
        instance13 = {"left_text": ["INCREDIBLY", "POOR", "SERVICE", "AN", 
                                    "FOOD", "QUALITY", "AT", "EXORBITANT"],
                      "target": ["PRICES"],
                      "right_text": ["@@EMPTY_SENTENCE@@"],
                      "epoch_number": [-1],
                      "sentiment": sentiment_mapper[-1]}
        test_instances = [instance1, instance5, instance12, instance13]
        
        if reverse_right_text:
            for test_instance in test_instances:
                test_instance['right_text'].reverse()

        if incl_target:
            for test_instance in test_instances:
                target = test_instance['target']
                left_text = test_instance['left_text']
                # Handle the empty sentence case
                if left_text == ["@@EMPTY_SENTENCE@@"]:
                    left_text = []
                left_text += target

                right_text = test_instance['right_text']
                if right_text == ["@@EMPTY_SENTENCE@@"]:
                    right_text = []

                if reverse_right_text:
                    target_temp = copy.deepcopy(target)
                    target_temp.reverse()
                    right_text += target_temp
                else:
                    right_text = target + right_text

                test_instance['left_text'] = left_text
                test_instance['right_text'] = right_text

        assert len(instances) == 13
        fields = instances[0].fields
        assert [t.text for t in fields["left_text"].tokens] == instance1["left_text"]
        assert [t.text for t in fields["right_text"].tokens] == instance1["right_text"]
        assert [t.text for t in fields["target"].tokens] == instance1["target"]
        assert fields["label"].label == instance1["sentiment"]
        if augmented_data:
            assert list(fields["epoch_numbers"].array) == instance1["epoch_number"]
        fields = instances[4].fields
        assert [t.text for t in fields["left_text"].tokens] == instance5["left_text"]
        assert [t.text for t in fields["right_text"].tokens] == instance5["right_text"]
        assert [t.text for t in fields["target"].tokens] == instance5["target"]
        assert fields["label"].label == instance5["sentiment"]
        if augmented_data:
            assert list(fields["epoch_numbers"].array) == instance5["epoch_number"]
        fields = instances[11].fields
        assert [t.text for t in fields["left_text"].tokens] == instance12["left_text"]
        assert [t.text for t in fields["right_text"].tokens] == instance12["right_text"]
        assert [t.text for t in fields["target"].tokens] == instance12["target"]
        assert fields["label"].label == instance12["sentiment"]
        if augmented_data:
            assert list(fields["epoch_numbers"].array) == instance12["epoch_number"]
        fields = instances[12].fields
        assert [t.text for t in fields["left_text"].tokens] == instance13["left_text"]
        assert [t.text for t in fields["right_text"].tokens] == instance13["right_text"]
        assert [t.text for t in fields["target"].tokens] == instance13["target"]
        assert fields["label"].label == instance13["sentiment"]
        if augmented_data:
            assert list(fields["epoch_numbers"].array) == instance13["epoch_number"]