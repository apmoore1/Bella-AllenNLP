import sys
from pathlib import Path
import os
sys.path.insert(0, str(Path('.').resolve()))
print(sys.path)

from bella_allen_nlp.dataset_readers.tdlstm import TDLSTMDatasetReader

from allennlp.common.params import Params
from allennlp.models import Model
from pathlib import Path
from allennlp.data import DataIterator, DatasetReader, Vocabulary
from allennlp.data.dataset import Batch

param_file = Path('/home/andrew/Documents/Bella-AllenNLP/tests/test_data/tclstm_model_config.json')
params = Params.from_file(param_file)

reader = DatasetReader.from_params(params['dataset_reader'])
# The dataset reader might be lazy, but a lazy list here breaks some of our tests.
instances = list(reader.read('/home/andrew/Documents/Bella-AllenNLP/tests/test_data/target_reader_data.json'))
# Use parameters for vocabulary if they are present in the config file, so that choices like
# "non_padded_namespaces", "min_count" etc. can be set if needed.
if 'vocabulary' in params:
    vocab_params = params['vocabulary']
    vocab = Vocabulary.from_params(params=vocab_params, instances=instances)
else:
    vocab = Vocabulary.from_instances(instances)
model = Model.from_params(vocab=vocab, params=params['model'])
dataset = Batch(instances)
dataset.index_instances(vocab)
tensors = dataset.as_tensor_dict(dataset.get_padding_lengths())
model(**tensors)
print('anything')