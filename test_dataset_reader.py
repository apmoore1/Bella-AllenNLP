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


param_fp = str(Path('tests', 'test_data', 'tclstm_model_config.json').resolve())
params = Params.from_file(param_fp).duplicate()
#import pdb
#pdb.set_trace()
model = Model.from_params(vocab=vocab, params=params['model'])

#a=model(**tensor_dict)
t_end = train_dataset[-1]
batch_alt = Batch([t_end])
batch_alt.index_instances(vocab)
padding_lengths_alt = batch_alt.get_padding_lengths()
tensor_dict_alt = batch_alt.as_tensor_dict(padding_lengths_alt)
a = model(**tensor_dict_alt)
import pdb
pdb.set_trace()

vocab = Vocabulary.from_instances(train_dataset + validation_dataset)
WORD_EMBEDDING_DIM = 50
CHAR_EMBEDDING_DIM = 5
CHAR_WORD_DIM = 30
HIDDEN_DIM = 50


#char_embedding = Embedding(num_embeddings=vocab.get_vocab_size("char_id"), 
#                           embedding_dim=CHAR_EMBEDDING_DIM)
#character_cnn = CnnEncoder(embedding_dim=CHAR_EMBEDDING_DIM, num_filters=2, 
#                           output_dim=CHAR_WORD_DIM)
#token_character_encoder = TokenCharactersEncoder(embedding=char_embedding, 
#                                                 encoder=character_cnn)

#word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding,
#                                          "chars": token_character_encoder})


# Model
glove_fp = cached_path('/home/andrew/glove.6B/glove.6B.50d.txt')
glove_50_weights = _read_pretrained_embeddings_file(glove_fp, 50, vocab, 'tokens_id')

token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens_id'),
                            embedding_dim=WORD_EMBEDDING_DIM,
                            weight=glove_50_weights)
#token_embedding1 = Embedding(num_embeddings=vocab.get_vocab_size('tokens_id'),
#                            embedding_dim=WORD_EMBEDDING_DIM)
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
                  num_epochs=40,
                  histogram_interval=100, should_log_learning_rate=True)

serialization_dir = '/tmp/anything100'
another_log = SummaryWriter(os.path.join(serialization_dir, "log", "embeddings"))
train_log = SummaryWriter(os.path.join(serialization_dir, "log", "train"))
validation_log = SummaryWriter(os.path.join(serialization_dir, "log", "validation"))


trainer._tensorboard = TensorboardWriter(train_log=train_log, validation_log=validation_log)

trainer.train()
# Project the learnt word embeddings
another_log.add_embedding(token_embedding.weight, metadata=token_names, 
                          tag='Sentiment Embeddings')
# Project the Original word embeddings
original_50_weights = _read_pretrained_embeddings_file(glove_fp, 50, vocab, 'tokens_id')
another_log.add_embedding(original_50_weights, metadata=token_names, 
                          tag='Original Embeddings')
train_log.close()
validation_log.close()
another_log.close()

#predictor = SentenceTaggerPredictor(model, dataset_reader=reader)
#tag_logits = predictor.predict("The dog ate the apple")['tag_logits']
#tag_ids = np.argmax(tag_logits, axis=-1)
#print([model.vocab.get_token_from_index(i, 'labels') for i in tag_ids])

# Some type of debugging
#for c in iterator(train_dataset):
#    print(c)
#    print(word_embeddings(c['text']))
#    word_embeds = word_embeddings(c['text'])
#     print(word_embeddings(c))
#    print('something')
#    break
#print('anything')
