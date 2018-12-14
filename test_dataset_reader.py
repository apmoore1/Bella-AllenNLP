from allennlp.common.file_utils import cached_path
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.token_indexers import TokenCharactersIndexer, SingleIdTokenIndexer
from allennlp.modules.token_embedders import Embedding, TokenCharactersEncoder
from allennlp.modules.token_embedders.embedding import _read_pretrained_embeddings_file
from allennlp.modules.seq2vec_encoders import CnnEncoder, PytorchSeq2VecWrapper
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.data.iterators import BucketIterator
from allennlp.training.trainer import Trainer, TensorboardWriter

from allennlp.models import Model
from allennlp.common.params import Params
from allennlp.data.dataset import Batch


from tensorboardX import SummaryWriter, FileWriter

import torch
import torch.optim as optim

import sys
from pathlib import Path
import os
sys.path.insert(0, str(Path('.').resolve()))
print(sys.path)

from bella_allen_nlp.dataset_readers.tdlstm import TDLSTMDatasetReader
from bella_allen_nlp.allen_models.tdlstm import TDLSTMClassifier
from bella.parsers import semeval_14

sem_dir = Path('..', '..', 'aspect datasets', 'semeval_2014')
laptop_train = semeval_14(Path(sem_dir, 'laptop_train.xml'), name='Laptop')
rest_train = semeval_14(Path(sem_dir, 'restaurants_train.xml'), 
                        name='Restaurant Train')
rest_test = semeval_14(Path(sem_dir, 'restaurants_test.xml'), 
                       name='Restaurant Test')

rest_train_fps = rest_train.to_json_file(['Restaurant Train', 'Restaurant Dev'], 0.2, 
                                         random_state=42)
rest_test_fp = rest_test.to_json_file('Restaurant Test')
laptop_fps = laptop_train.to_json_file(['Laptop Train', 'Laptop Dev'], 0.2, 
                                       random_state=42)
rest_train_fp, rest_dev_fp = rest_train_fps
laptop_train_fp, laptop_dev_fp = laptop_fps
import pdb
pdb.set_trace()
#token_indexers = {'tokens': SingleIdTokenIndexer(namespace='tokens_id'),
#                  'chars': TokenCharactersIndexer(namespace='char_id')}
token_indexers = {'tokens': SingleIdTokenIndexer(namespace='tokens', 
                                                 lowercase_tokens=True)}
reader = TDLSTMDatasetReader(token_indexers=token_indexers, incl_target=False)
data = str(Path('tests', 'test_data', 'target_reader_data.json').resolve())
train_dataset = reader.read(cached_path(data))

vocab = Vocabulary().from_instances(train_dataset)
#target = train_dataset[0].fields['target']
#text = train_dataset[0].fields['text']
#label = train_dataset[0].fields['label']
t_1 = train_dataset[0]
t_2 = train_dataset[1]
batch = Batch([t_1,t_2])
#vocab = Vocabulary.from_instances(batch)
batch.index_instances(vocab)
padding_lengths = batch.get_padding_lengths()
tensor_dict = batch.as_tensor_dict(padding_lengths)