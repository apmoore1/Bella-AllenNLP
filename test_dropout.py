import sys
from pathlib import Path
import os
sys.path.insert(0, str(Path('..').resolve()))
import logging

from allennlp.common.file_utils import cached_path
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.token_indexers import TokenCharactersIndexer, SingleIdTokenIndexer
from allennlp.modules.token_embedders import Embedding, TokenCharactersEncoder
from allennlp.modules.token_embedders.embedding import _read_pretrained_embeddings_file
from allennlp.modules.seq2vec_encoders import CnnEncoder, PytorchSeq2VecWrapper
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.data.iterators import BucketIterator
from allennlp.training.trainer import Trainer, TensorboardWriter

from bella.parsers import semeval_14

import torch
import torch.optim as optim


from allennlp.commands.train import train_model_from_file
from allennlp.common import Params
from allennlp.common.testing.test_case import AllenNlpTestCase
from allennlp.data import DataIterator, DatasetReader, Vocabulary
from allennlp.data.dataset import Batch
from allennlp.models import Model, load_archive

from bella_allen_nlp.dataset_readers.target import TargetDatasetReader
from bella_allen_nlp.allen_models.target import TargetClassifier

import tempfile

logging.basicConfig(format='%(message)s',
                    level=logging.INFO)


model_fp = str(Path('.', 'test_dropout.json').resolve())
params = Params.from_file(model_fp)
data_fp = str(Path('tests', 'test_data', 'target_reader_data.json').resolve())
reader = DatasetReader.from_params(params['dataset_reader'])
instances = list(reader.read(data_fp))
if 'vocabulary' in params:
    vocab_params = params['vocabulary']
    vocab = Vocabulary.from_params(params=vocab_params, instances=instances)
else:
    vocab = Vocabulary.from_instances(instances)
model = Model.from_params(vocab=vocab, params=params['model'])

with tempfile.TemporaryDirectory() as tmpdirname:
    model = train_model_from_file(model_fp, tmpdirname,overrides="")

print('anything')
# Model
glove_fp = cached_path('/home/andrew/glove.6B/glove.6B.50d.txt')
glove_50_weights = _read_pretrained_embeddings_file(glove_fp, 50, vocab, 'tokens_id')

token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens_id'),
                            embedding_dim=WORD_EMBEDDING_DIM,
                            weight=glove_50_weights)

id_to_tokens = vocab.get_index_to_token_vocabulary(namespace='tokens_id')
token_names = list(id_to_tokens.values())

word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})

text_lstm = PytorchSeq2VecWrapper(torch.nn.LSTM(WORD_EMBEDDING_DIM, HIDDEN_DIM, batch_first=True))
target_lstm = PytorchSeq2VecWrapper(torch.nn.LSTM(WORD_EMBEDDING_DIM, HIDDEN_DIM, batch_first=True))
feed_forward = torch.nn.Linear(HIDDEN_DIM * 2,
                               out_features=vocab.get_vocab_size('labels'))

model = TargetLSTMClassifier(vocab, word_embeddings, text_lstm, target_lstm, feed_forward)

# Data iterator
sort_fields = [("text", "num_tokens"), ("target", "num_tokens")]
iterator = BucketIterator(batch_size=32, sorting_keys=sort_fields)
iterator.index_with(vocab)

# Model training
optimizer = optim.Adam(model.parameters())

trainer = Trainer(model=model,
                  optimizer=optimizer,
                  iterator=iterator,
                  train_dataset=train_dataset,
                  validation_dataset=validation_dataset,
                  patience=10,
                  num_epochs=1)

trainer.train()

