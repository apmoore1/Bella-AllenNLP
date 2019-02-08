from pathlib import Path

from allennlp.common.util import ensure_list
import pytest

from bella_allen_nlp.dataset_readers.target import TargetDatasetReader

class TestTargetDatasetReader():
    @pytest.mark.parametrize("augmented_data", (True, False))
    @pytest.mark.parametrize("sentiment_mapper", [None, {-1: 'Neg', 0: 'Neu', 
                                                         1: 'Pos'}])
    @pytest.mark.parametrize("lazy", (True, False))
    def test_read_from_file(self, lazy, sentiment_mapper, augmented_data):
        reader = TargetDatasetReader(lazy=lazy,
                                     sentiment_mapper=sentiment_mapper)
        if sentiment_mapper is None:
            sentiment_mapper = {-1: 'negative', 0: 'neutral', 1: 'positive'}
        test_fp = Path(__file__, '..', '..', 'test_data', 'data', 
                       'target_reader_data.json')
        if augmented_data:
            test_fp = Path(__file__, '..', '..', 'test_data', 'data', 
                           'augmented_target_reader_data.json')
        instances = ensure_list(reader.read(str(test_fp.resolve())))

        instance1 = {"text": ["Though", "you", "will", "undoubtedly", "be", 
                              "seated", "at", "a"],
                     "target": ["spot"], "epoch_number": [-1],
                     "sentiment": sentiment_mapper[-1]}
        instance2 = {"text": ["The", "bar", "is", "very", "well", "stocked"],
                     "target": ["bar"], "epoch_number": [-1],
                     "sentiment": sentiment_mapper[1]}
        instance3 = {"text": ["Even", "after", "getting", 
                              "pushed", "out", "by"],
                     "target": ["Pizza"], "epoch_number": [2, 5, 9],
                     "sentiment": sentiment_mapper[1]}
        instance4 = {"text": ["Good", ",", "because", "hey", ",", "it"],
                     "target": ["dishes"], "epoch_number": [8],
                     "sentiment": sentiment_mapper[0]}

        assert len(instances) == 13
        fields = instances[0].fields
        assert [t.text for t in fields["text"].tokens[:8]] == instance1["text"]
        assert [t.text for t in fields["target"].tokens] == instance1["target"]
        assert fields["label"].label == instance1["sentiment"]
        if augmented_data:
            assert list(fields["epoch_numbers"].array) == instance1["epoch_number"]
        fields = instances[1].fields
        assert [t.text for t in fields["text"].tokens[:6]] == instance2["text"]
        assert [t.text for t in fields["target"].tokens[:1]] == instance2["target"]
        assert fields["label"].label == instance2["sentiment"]
        if augmented_data:
            assert list(fields["epoch_numbers"].array) == instance2["epoch_number"]
        fields = instances[2].fields
        assert [t.text for t in fields["text"].tokens[:6]] == instance3["text"]
        assert [t.text for t in fields["target"].tokens] == instance3["target"]
        assert fields["label"].label == instance3["sentiment"]
        if augmented_data:
            assert list(fields["epoch_numbers"].array) == instance3["epoch_number"]
        fields = instances[3].fields
        assert [t.text for t in fields["text"].tokens[:6]] == instance4["text"]
        assert [t.text for t in fields["target"].tokens] == instance4["target"]
        assert fields["label"].label == instance4["sentiment"]
        if augmented_data:
            assert list(fields["epoch_numbers"].array) == instance4["epoch_number"]