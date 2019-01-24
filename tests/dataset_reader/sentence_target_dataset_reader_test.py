from pathlib import Path

from allennlp.common.util import ensure_list
import pytest

from bella_allen_nlp.dataset_readers.sentence_target import SentenceTargetDatasetReader

class TestSentenceTargetDatasetReader():
    @pytest.mark.parametrize("sentiment_mapper", [None, {-1: 'Neg', 0: 'Neu', 
                                                         1: 'Pos'}])
    @pytest.mark.parametrize("lazy", (True, False))
    def test_read_from_file(self, lazy, sentiment_mapper):
        reader = SentenceTargetDatasetReader(lazy=lazy, 
                                             sentiment_mapper=sentiment_mapper)
        if sentiment_mapper is None:
            sentiment_mapper = {-1: 'negative', 0: 'neutral', 1: 'positive'}
        test_fp = Path(__file__, '..', '..', 'test_data', 'data', 
                       'sentence_target_reader_data.json')
        instances = ensure_list(reader.read(str(test_fp.resolve())))

        instance1 = {"text": ["Word", "is", "a", "great", "program", 
                              "for", "storing", "and", "organizing", "files", 
                              "."],
                     "targets": [["program"], ["Word"]],
                     "sentiments": [sentiment_mapper[1], sentiment_mapper[1]]}
        instance2 = {"text": ["The", "bar", "is", "very", "well", "stocked"],
                     "target": ["bar"],
                     "sentiment": sentiment_mapper[1]}
        instance3 = {"text": ["Even", "after", "getting", 
                              "pushed", "out", "by"],
                     "target": ["Pizza"],
                     "sentiment": sentiment_mapper[1]}

        assert len(instances) == 3
        fields = instances[0].fields
        assert [t.text for t in fields["text"]] == instance1["text"]
        for index, target_field in enumerate(fields['targets']):
            assert [t.text for t in target_field] == instance1["targets"][index]
        for index, sentiment_field in enumerate(fields['labels']):
            assert sentiment_field.label == instance1["sentiments"][index]
        #assert fields["label"].label == instance1["sentiment"]
        #fields = instances[1].fields
        #assert [t.text for t in fields["text"].tokens[:6]] == instance2["text"]
        #assert [t.text for t in fields["target"].tokens[:1]] == instance2["target"]
        #assert fields["label"].label == instance2["sentiment"]
        #fields = instances[2].fields
        #assert [t.text for t in fields["text"].tokens[:6]] == instance3["text"]
        #assert [t.text for t in fields["target"].tokens] == instance3["target"]
        #assert fields["label"].label == instance3["sentiment"]
        #fields = instances[3].fields
        #assert [t.text for t in fields["text"].tokens[:6]] == instance4["text"]
        #assert [t.text for t in fields["target"].tokens] == instance4["target"]
        #assert fields["label"].label == instance4["sentiment"]