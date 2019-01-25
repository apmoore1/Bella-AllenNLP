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
        instance2 = {"text": ["The", "So", "called", "desktop", "Runs", "to",
                              "badly"],
                     "targets": [["Runs"]],
                     "sentiments": [sentiment_mapper[-1]]}
        instance3 = {"text": ["errrr", "when", "I", "order", "I", "did", "go",
                              "full", "scale", "for", "the", "noting", "or",
                              "full", "trackpad", "I", "wanted", "something",
                              "for", "basics", "of", "being", "easy", "to",
                              "move", "It", "."],
                     "targets": [["noting"], ["move", "It"], ["trackpad"]],
                     "sentiments": [sentiment_mapper[0], sentiment_mapper[1], 
                                    sentiment_mapper[0]]}

        assert len(instances) == 3
        fields = instances[0].fields
        assert [t.text for t in fields["text"]] == instance1["text"]
        for index, target_field in enumerate(fields['targets']):
            assert [t.text for t in target_field] == instance1["targets"][index]
        assert fields['labels'].labels == instance1["sentiments"]
        
        fields = instances[1].fields
        assert [t.text for t in fields["text"]] == instance2["text"]
        for index, target_field in enumerate(fields['targets']):
            assert [t.text for t in target_field] == instance2["targets"][index]
        assert fields['labels'].labels == instance2["sentiments"]

        fields = instances[2].fields
        assert [t.text for t in fields["text"]] == instance3["text"]
        for index, target_field in enumerate(fields['targets']):
            assert [t.text for t in target_field] == instance3["targets"][index]
        assert fields['labels'].labels == instance3["sentiments"]