from typing import Dict, Optional

from allennlp.common.checks import check_dimensions_match
from allennlp.data import Vocabulary
from allennlp.modules import FeedForward, Seq2SeqEncoder, TextFieldEmbedder, Seq2VecEncoder, TimeDistributed
from allennlp.modules import InputVariationalDropout
from allennlp.modules.attention import DotProductAttention
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn import util, Activation
from allennlp.training.metrics import CategoricalAccuracy, F1Measure
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules import Dropout, Dropout2d, Linear


@Model.register("attention_target_classifier")
class AttentionTargetClassifier(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 text_encoder: Seq2SeqEncoder,
                 target_encoder: Seq2VecEncoder,
                 feedforward: Optional[FeedForward] = None,
                 target_field_embedder: Optional[TextFieldEmbedder] = None,
                 target_concat_text_embedding: bool = False,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None,
                 word_dropout: float = 0.0,
                 variational_dropout: float = 0.0,
                 dropout: float = 0.0) -> None:
        '''
        :param vocab: vocab : A Vocabulary, required in order to compute sizes 
                              for input/output projections.
        :param text_field_embedder: Used to embed the text and target text if
                                    target_field_embedder is None but the 
                                    target_encoder is not None.
        :param text_encoder: Sequence Encoder that will create the 
                             representation of each token in the context 
                             sentence.
        :param target_encoder: Encoder that will create the representation of 
                               target text tokens.
        :param feedforward: An optional feed forward layer to apply after
                            either the text encoder if target encoder is None. 
                            Else it would be after the target and the text 
                            encoded representations have been concatenated.
        :param target_field_embedder: Used to embed the target text to give as 
                                      input to the target_encoder. Thus this 
                                      allows a seperate embedding for text and 
                                      target text.
        :param target_concat_text_embedding: Whether or not the target should be 
                                             concatenated to the each word 
                                             embedding within the text before 
                                             being encoded.
        :param initializer: Used to initialize the model parameters.
        :param regularizer: If provided, will be used to calculate the 
                            regularization penalty during training.
        :param word_dropout: Dropout that is applied after the embedding layer 
                             but before the variational_dropout. It will drop 
                             entire word/timesteps with the specified 
                             probability.
        :param variational_dropout: Dropout that is applied after a layer that 
                                    outputs a sequence of vectors. In this case 
                                    this is applied after the embedding layer 
                                    and the encoding of the text. This will 
                                    apply the same dropout mask to each 
                                    timestep compared to standard dropout 
                                    which would use a different dropout mask 
                                    for each timestep. Specify here the 
                                    probability of dropout.
        :param dropout: Standard dropout, applied to any output vector which 
                        is after the target encoding and the attention layer.
                        Specify here the probability of dropout.
        
        This attention method does not use Bi-Linear attention rather a 
        slightly different type. This class is all based around the following 
        paper `Attention-based LSTM for Aspect-level Sentiment Classification 
        <https://www.aclweb.org/anthology/D16-1058>`_. The default model here 
        is the equivalent to the AT-LSTM within this paper. If the 
        `target_concat_text_embedding` argument is `True` then the model becomes 
        the ATAE-LSTM within the cited paper.

        The only difference between this model and the attention based models 
        in the paper is that the final sentence representation is `r` rather 
        than `h* = tanh(Wpr + WxhN)` as we found this projection to not help 
        the performance.
        '''
        super().__init__(vocab, regularizer)

        self.text_field_embedder = text_field_embedder
        self.target_field_embedder = target_field_embedder
        self.num_classes = self.vocab.get_vocab_size("labels")
        self.text_encoder = text_encoder
        self.target_encoder = target_encoder
        self.feedforward = feedforward
        
        target_text_encoder_dim = (target_encoder.get_output_dim() + 
                                   text_encoder.get_output_dim())
        self.encoded_target_text_fusion = TimeDistributed(Linear(target_text_encoder_dim, 
                                                                 target_text_encoder_dim))
        self.attention_vector = Parameter(torch.Tensor(target_text_encoder_dim))
        self.attention_layer = DotProductAttention(normalize=True)

        if feedforward is not None:
            output_dim = self.feedforward.get_output_dim()
        else:
            output_dim = text_encoder.get_output_dim()
        self.label_projection = Linear(output_dim, self.num_classes)
        self.metrics = {
                "accuracy": CategoricalAccuracy()
        }
        self.f1_metrics = {}
        # F1 Scores
        label_index_name = self.vocab.get_index_to_token_vocabulary('labels')
        for label_index, label_name in label_index_name.items():
            label_name = f'F1_{label_name.capitalize()}'
            self.f1_metrics[label_name] = F1Measure(label_index)

        self._word_dropout = Dropout2d(word_dropout)
        self._variational_dropout = InputVariationalDropout(variational_dropout)
        self._naive_dropout = Dropout(dropout)

        self.target_concat_text_embedding = target_concat_text_embedding
        self.loss = torch.nn.CrossEntropyLoss()
        
        # Ensure the text encoder has the correct input dimension
        if target_concat_text_embedding:
            text_encoder_expected_in = (text_field_embedder.get_output_dim() + 
                                        target_encoder.get_output_dim())
            check_dimensions_match(text_encoder_expected_in, 
                                   text_encoder.get_input_dim(),
                                   "text field embedding dim + target encoder output dim", 
                                   "text encoder input dim")
        else:
            check_dimensions_match(text_field_embedder.get_output_dim(), 
                                   text_encoder.get_input_dim(),
                                   "text field embedding dim", 
                                   "text encoder input dim")
        # Ensure that the dimensions of the target or text field embedder and 
        # the target encoder match
        target_field_embedder_dim = text_field_embedder.get_output_dim()
        target_field_error = "text field embedding dim"
        if self.target_field_embedder:
            target_field_embedder_dim = target_field_embedder.get_output_dim()
            target_field_error = "target field embedding dim"
        
        check_dimensions_match(target_field_embedder_dim, 
                                target_encoder.get_input_dim(),
                                target_field_error, "target encoder input dim")
        self.reset_parameters()
        initializer(self)

    def reset_parameters(self):
        '''
        Intitalises the attnention vector
        '''
        torch.nn.init.uniform_(self.attention_vector, -0.01, 0.01)

    def _token_dropout(self, embedded_text: torch.FloatTensor
                      ) -> torch.FloatTensor:
        '''
        Dropout will randomly drop whole words.

        This is equivalent to `1D Spatial Dropout`_. 
        
        .. _1D Spatial 
           Dropout:https://keras.io/layers/core/#spatialdropout1d 

        :param embedded_text: A tensor of shape: 
                              [batch_size, timestep, embedding_dim] of which 
                              the dropout will drop entire timestep which is 
                              the equivalent to words.
        :returns: The given tensor but with timesteps/words dropped.
        '''
        embedded_text = embedded_text.unsqueeze(2)
        embedded_text = self._word_dropout(embedded_text)
        return embedded_text.squeeze(2)
        
    def forward(self,
                text: Dict[str, torch.LongTensor],
                target: Dict[str, torch.LongTensor],
                label: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
        '''
        The text and targets are Dictionaries as they are text fields they can 
        be represented many different ways e.g. just words or words and chars 
        etc therefore the dictionary represents these different ways e.g. 
        {'words': words_tensor_ids, 'chars': char_tensor_ids}
        '''
        # Embed and encode target as a vector
        if self.target_field_embedder:
            embedded_target = self.target_field_embedder(target)
        else:
            embedded_target = self.text_field_embedder(target)
        embedded_target = self._token_dropout(embedded_target)
        embedded_target = self._variational_dropout(embedded_target)
        target_mask = util.get_text_field_mask(target)
        encoded_target = self.target_encoder(embedded_target, target_mask)
        
        # Embed text
        embedded_text = self.text_field_embedder(text)
        #embedded_text = self._token_dropout(embedded_text)
        text_mask = util.get_text_field_mask(text)
        


        # Encoded target to be of dimension (batch, words, dim) currently
        # (batch, dim), this is so that we can concat the target and encoded 
        # text
        encoded_target = encoded_target.unsqueeze(1)
        # Need to repeat the target word for each word in context text
        text_num_padded = embedded_text.shape[1]
        encoded_targets = encoded_target.repeat((1, text_num_padded, 1))
        # If it is the ATAE method then this needs to be done for the AE part
        if self.target_concat_text_embedding:
            embedded_text = torch.cat((embedded_text, encoded_targets), -1)
        embedded_text = self._token_dropout(embedded_text)
        embedded_text = self._variational_dropout(embedded_text)

        # Encode text sequence
        encoded_text_seq = self.text_encoder(embedded_text, text_mask)
        # get the last sequence (final hidden states) of the encoded text
        index_of_last_sequence = text_mask.sum(1) - 1
        batch_size, seq_len = text_mask.shape
        index_multi_batch = (torch.arange(0, batch_size) * seq_len) + index_of_last_sequence
        flattened_encoded_text_seq = encoded_text_seq.view(batch_size * seq_len, -1)
        encoded_text_final_states = torch.index_select(flattened_encoded_text_seq, 
                                                       0, index_multi_batch)
        encoded_text_final_states = self._naive_dropout(encoded_text_final_states)

        # Join the encoded text and targets together
        target_encoded_text = torch.cat((encoded_text_seq, encoded_targets), -1)
        target_encoded_text = self._variational_dropout(target_encoded_text)

        # Target and text infusion layer
        infused_target_encoded_text = self.encoded_target_text_fusion(target_encoded_text)
        infused_target_encoded_text = torch.tanh(infused_target_encoded_text)
        # Attention based on a context vector which is to find the most informative 
        batch_size = text_mask.shape[0]
        attention_vector = self.attention_vector.unsqueeze(0).expand(batch_size, -1)
        attention_weights = self.attention_layer(attention_vector, 
                                                 infused_target_encoded_text, 
                                                 text_mask)
        attention_weights = attention_weights.unsqueeze(-1)
        weighted_encoded_text_seq = encoded_text_seq * attention_weights
        weighted_encoded_text_vec = weighted_encoded_text_seq.sum(1)
        weighted_encoded_text_vec = self._naive_dropout(weighted_encoded_text_vec)

        if self.feedforward:
            weighted_encoded_text_vec = self.feedforward(weighted_encoded_text_vec)
        logits = self.label_projection(weighted_encoded_text_vec)
        class_probabilities = F.softmax(logits, dim=-1)

        output_dict = {"class_probabilities": class_probabilities}

        if label is not None:
            loss = self.loss(logits, label)
            for metrics in [self.metrics, self.f1_metrics]:
                for metric in metrics.values():
                    metric(logits, label)
            output_dict["loss"] = loss

        return output_dict

    def decode(self, output_dict: Dict[str, torch.Tensor]
               ) -> Dict[str, torch.Tensor]:
        predictions = output_dict['class_probabilities'].cpu().data.numpy()
        argmax_indices = np.argmax(predictions, axis=-1)
        labels = [self.vocab.get_token_from_index(x, namespace="labels")
                  for x in argmax_indices]
        output_dict['label'] = labels
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        # Other scores
        metric_name_value = {}
        for metric_name, metric in self.metrics.items():
            metric_name_value[metric_name] = metric.get_metric(reset)
        # F1 scores
        all_f1_scores = []
        for metric_name, metric in self.f1_metrics.items():
            precision, recall, f1_measure = metric.get_metric(reset)
            all_f1_scores.append(f1_measure)
            metric_name_value[metric_name] = f1_measure
        metric_name_value['Macro_F1'] = sum(all_f1_scores) / len(self.f1_metrics)
        return metric_name_value