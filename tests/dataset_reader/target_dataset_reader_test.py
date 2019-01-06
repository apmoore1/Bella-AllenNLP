from pathlib import Path

from allennlp.common.util import ensure_list
import pytest

from bella_allen_nlp.dataset_readers.target import TargetDatasetReader

class TestTargetDatasetReader():
    @pytest.mark.parametrize("lazy", (True, False))
    def test_read_from_file(self, lazy):
        reader = TargetDatasetReader(lazy=lazy)
        test_fp = Path(__file__, '..', '..', 'test_data', 
                       'target_reader_data.json')
        instances = ensure_list(reader.read(str(test_fp.resolve())))

        instance1 = {"text": ["Though", "you", "will", "undoubtedly", "be", 
                              "seated", "at", "a"],
                     "target": ["spot"],
                     "sentiment": 'negative'}
        instance2 = {"text": ["The", "bar", "is", "very", "well", "stocked"],
                     "target": ["bar"],
                     "sentiment": 'positive'}
        instance3 = {"text": ["Even", "after", "getting", 
                              "pushed", "out", "by"],
                     "target": ["Pizza"],
                     "sentiment": 'positive'}
        instance4 = {"text": ["Good", ",", "because", "hey", ",", "it"],
                     "target": ["dishes"],
                     "sentiment": "neutral"}

        assert len(instances) == 13
        fields = instances[0].fields
        assert [t.text for t in fields["text"].tokens[:8]] == instance1["text"]
        assert [t.text for t in fields["target"].tokens] == instance1["target"]
        assert fields["label"].label == instance1["sentiment"]
        fields = instances[1].fields
        assert [t.text for t in fields["text"].tokens[:6]] == instance2["text"]
        assert [t.text for t in fields["target"].tokens[:1]] == instance2["target"]
        assert fields["label"].label == instance2["sentiment"]
        fields = instances[2].fields
        assert [t.text for t in fields["text"].tokens[:6]] == instance3["text"]
        assert [t.text for t in fields["target"].tokens] == instance3["target"]
        assert fields["label"].label == instance3["sentiment"]
        fields = instances[3].fields
        assert [t.text for t in fields["text"].tokens[:6]] == instance4["text"]
        assert [t.text for t in fields["target"].tokens] == instance4["target"]
        assert fields["label"].label == instance4["sentiment"]