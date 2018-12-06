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


from bella_allen_nlp.dataset_readers.target import TargetDatasetReader
from bella_allen_nlp.allen_models.target_lstm import TargetLSTMClassifier

logging.basicConfig(format='%(message)s',
                    level=logging.INFO)

sem_dir = Path('..', '..', 'aspect datasets', 'semeval_2014')
laptop_train = semeval_14(Path(sem_dir, 'laptop_train.xml'), name='Laptop')
rest_train = semeval_14(Path(sem_dir, 'restaurants_train.xml'), 
                        name='Restaurant')
rest_fps = rest_train.to_json_file(['Restaurant Train', 'Restaurant Dev'], 0.2, 
                                   random_state=42)
laptop_fps = laptop_train.to_json_file(['Laptop Train', 'Laptop Dev'], 0.2, 
                                       random_state=42)
rest_train_fp, rest_dev_fp = rest_fps
laptop_train_fp, laptop_dev_fp = laptop_fps


token_indexers = {'tokens': SingleIdTokenIndexer(namespace='tokens_id', 
                                                 lowercase_tokens=True)}
reader = TargetDatasetReader(token_indexers=token_indexers)
train_dataset = reader.read(cached_path(rest_train_fp))
validation_dataset = reader.read(cached_path(rest_dev_fp))
target = train_dataset[0].fields['target']
text = train_dataset[0].fields['text']
label = train_dataset[0].fields['label']

vocab = Vocabulary.from_instances(train_dataset + validation_dataset)
WORD_EMBEDDING_DIM = 50
CHAR_EMBEDDING_DIM = 5
CHAR_WORD_DIM = 30
HIDDEN_DIM = 50


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

