from typing import Dict, Optional

from allennlp.common.checks import check_dimensions_match
from allennlp.data import Vocabulary
from allennlp.modules import FeedForward, Seq2SeqEncoder, TextFieldEmbedder
from allennlp.modules import InputVariationalDropout, Seq2VecEncoder, TimeDistributed
from allennlp.modules.matrix_attention import BilinearMatrixAttention
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn import util, Activation
from allennlp.training.metrics import CategoricalAccuracy, F1Measure
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules import Dropout, Dropout2d, Linear


@Model.register("attention_multi_target_classifier")
class AttentionMultiTargetClassifier(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 text_encoder: Seq2SeqEncoder,
                 target_encoder: Seq2VecEncoder,
                 feedforward: Optional[FeedForward] = None,
                 target_field_embedder: Optional[TextFieldEmbedder] = None,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None,
                 dropout: float = 0.0,
                 target_scale: bool = False,
                 context_preserving: bool = False) -> None:
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
        :param word_dropout: Dropout that is applied after the embedding of the 
                             tokens/words. It will drop entire words with this 
                             probabilty.
        :param dropout: To apply dropout after each layer apart from the last 
                        layer. All dropout that is applied to timebased data 
                        will be `variational dropout`_ all else will be  
                        standard dropout.

        .. _variational dropout:
           https://papers.nips.cc/paper/6241-a-theoretically-grounded-application-of-dropout-in-recurrent-neural-networks.pdf
        '''
        super().__init__(vocab, regularizer)

        self.text_field_embedder = text_field_embedder
        self.target_field_embedder = target_field_embedder
        self.num_classes = self.vocab.get_vocab_size("labels")
        self.text_encoder = text_encoder
        self.target_encoder = TimeDistributed(target_encoder)
        self.feedforward = feedforward
        if self.feedforward:
            self.time_feedforward = TimeDistributed(self.feedforward)
        
        self.attention_layer = BilinearMatrixAttention(text_encoder.get_output_dim(),
                                                       target_encoder.get_output_dim())
        # Whether to concat the encoded text representation with the weighted 
        # representation from the attention
        self.context_preserving = context_preserving

        if feedforward is not None:
            output_dim = self.feedforward.get_output_dim()
        else:
            if self.context_preserving:
                output_dim = (text_encoder.get_output_dim() * 2)
            else:
                output_dim = text_encoder.get_output_dim()
        self.label_projection = TimeDistributed(Linear(output_dim, self.num_classes))
        self.metrics = {
                "accuracy": CategoricalAccuracy()
        }
        self.f1_metrics = {}
        # F1 Scores
        label_index_name = self.vocab.get_index_to_token_vocabulary('labels')
        for label_index, label_name in label_index_name.items():
            label_name = f'F1_{label_name.capitalize()}'
            self.f1_metrics[label_name] = F1Measure(label_index)

        self._variational_dropout = InputVariationalDropout(dropout)
        self._naive_dropout = Dropout(dropout)
        self._time_naive_dropout = TimeDistributed(self._naive_dropout)
        self._time_variational_dropout = TimeDistributed(self._variational_dropout)

        self.target_scale = target_scale

        self.loss = torch.nn.CrossEntropyLoss()
        
        
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
        initializer(self)
        
    def forward(self,
                text: Dict[str, torch.LongTensor],
                targets: Dict[str, torch.LongTensor],
                labels: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
        '''
        The text and targets are Dictionaries as they are text fields they can 
        be represented many different ways e.g. just words or words and chars 
        etc therefore the dictionary represents these different ways e.g. 
        {'words': words_tensor_ids, 'chars': char_tensor_ids}
        '''
        # The targets are going to be of shape: 
        # (batch_size, number of targets in sentence, num_tokens)
        # Labels will be of shape (batch_size, number of targets in sentence)
        # As the number of targets can be padded this has to be taken into
        # account by checking if the target is fully padded or not.
        # Embed and encode target as a vector
        if self.target_field_embedder:
            embedded_target = self.target_field_embedder(targets)
        else:
            embedded_target = self.text_field_embedder(targets)
        embedded_target = self._time_variational_dropout(embedded_target)
        
        # Mask for the target text
        # (batch_size, number of targets, num_tokens)
        targets_mask = util.get_text_field_mask(targets)
        # Mask to enough how many targets there are per text, used to also 
        # know how many labels are associated to one text
        # (batch_size, number of targets)
        target_text_mask = util.get_text_field_mask(targets, num_wrapping_dims=1)
        # (batch_size, number of targets, target encoder out dim)
        encoded_target = self.target_encoder(embedded_target, target_text_mask)
        encoded_target = self._time_naive_dropout(encoded_target)

        text_mask = util.get_text_field_mask(text)
        
        # Embed text
        embedded_text = self.text_field_embedder(text)
        embedded_text = self._variational_dropout(embedded_text)
        text_mask = util.get_text_field_mask(text)
        encoded_text_seq = self.text_encoder(embedded_text, text_mask)
        encoded_text_seq = self._variational_dropout(encoded_text_seq)

        # These are the similarties between each word and each target
        # (batch_size, text_length, number of targets)
        attention_weights = self.attention_layer(encoded_text_seq, encoded_target)
        # Masks for the targets but not the text
        softmax_attention_weights = util.masked_softmax(attention_weights, targets_mask)
        # This scales the weights so that the maximum is 1 rather than 1 / num targets
        if self.target_scale:
            num_targets_vec = targets_mask.sum(1).unsqueeze(-1).unsqueeze(-1)
            num_targets_vec = num_targets_vec.expand_as(softmax_attention_weights)
            softmax_attention_weights = num_targets_vec.float() * softmax_attention_weights
        # Multiply the softmax attention weights by the text sequence to get 
        # (batch size, sequence length, number targets, text encoding dim out)
        softmax_attention_weights = softmax_attention_weights.unsqueeze(-1)
        weighted_encoded_text_seq = encoded_text_seq.unsqueeze(2)
        weighted_encoded_text_seq = softmax_attention_weights * weighted_encoded_text_seq 
        # (batch_size, number targets, sequence length, text encoding dim out)
        weighted_encoded_text_seq = weighted_encoded_text_seq.transpose(1,2)
        # Concats the original encoded text representation with the weighted
        if self.context_preserving:
            expanded_encoded_text_seq = encoded_text_seq.unsqueeze(1)
            expanded_encoded_text_seq = expanded_encoded_text_seq.expand_as(weighted_encoded_text_seq)
            weighted_encoded_text_seq = torch.cat((weighted_encoded_text_seq, 
                                                   expanded_encoded_text_seq), -1)
        # (batch_size, number targets, text encoding dim out)
        weighted_encoded_text_vec = weighted_encoded_text_seq.sum(2)
        weighted_encoded_text_vec = self._time_naive_dropout(weighted_encoded_text_vec)

        if self.feedforward:
            weighted_encoded_text_vec = self.time_feedforward(weighted_encoded_text_vec)
        logits = self.label_projection(weighted_encoded_text_vec)
        # Mask the targets that do not exist (only exist for padding)
        masked_class_probabilities = util.masked_softmax(logits, targets_mask.unsqueeze(-1))
        output_dict = {"masked_class_probabilities": masked_class_probabilities}

        if labels is not None:
            loss = util.sequence_cross_entropy_with_logits(logits, labels, 
                                                           targets_mask)
            for metrics in [self.metrics, self.f1_metrics]:
                for metric in metrics.values():
                    metric(logits, labels, targets_mask)
            output_dict["loss"] = loss
        return output_dict

    def decode(self, output_dict: Dict[str, torch.Tensor]
               ) -> Dict[str, torch.Tensor]:
        predictions = output_dict['masked_class_probabilities'].cpu().data.numpy()
        predictions_list = [predictions[i] for i in range(predictions.shape[0])]
        labels = []
        for predictions in predictions_list:
            for prediction in predictions:
                # If all the probabilites are zero then it is a masked target
                if prediction.sum() == 0:
                    continue
                label_index = np.argmax(prediction)
                label = self.vocab.get_token_from_index(label_index, 
                                                        namespace="labels")
                labels.append(label)
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