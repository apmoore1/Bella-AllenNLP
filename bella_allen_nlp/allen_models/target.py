from typing import Dict, Optional

from allennlp.common.checks import check_dimensions_match
from allennlp.data import Vocabulary
from allennlp.modules import FeedForward, Seq2VecEncoder, TextFieldEmbedder
from allennlp.modules import InputVariationalDropout
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn import util
from allennlp.training.metrics import CategoricalAccuracy, F1Measure
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.modules import Dropout, Dropout2d, Linear


@Model.register("target_classifier")
class TargetClassifier(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 text_encoder: Seq2VecEncoder,
                 feedforward: Optional[FeedForward] = None,
                 target_field_embedder: Optional[TextFieldEmbedder] = None,
                 target_encoder: Optional[Seq2VecEncoder] = None,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None,
                 word_dropout: float = 0.0,
                 dropout: float = 0.0) -> None:
        '''
        :param vocab: vocab : A Vocabulary, required in order to compute sizes 
                              for input/output projections.
        :param text_field_embedder: Used to embed the text and target text if
                                    target_field_embedder is None but the 
                                    target_encoder is not None.
        :param text_encoder: Encoder that will create the representation for 
                             the whole text.
        :param feedforward: An optional feed forward layer to apply after
                            either the text encoder if target encoder is None. 
                            Else it would be after the target and the text 
                            encoded representations have been concatenated.
        :param target_field_embedder: Used to embed the target text to give as 
                                      input to the target_encoder. Thus this 
                                      allows a seperate embedding for text and 
                                      target text.
        :param target_encoder: Encoder that will create the representation of 
                               target text tokens.
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
        
        If no target encoder is given, the text is encoded and optionaly put 
        through the feed forward layer(s). If the target encoder is specified 
        then the text encoded representation is concatenated with the target 
        encoded representation and then optionaly put through the feed forward 
        layer(s).
        '''
        super().__init__(vocab, regularizer)

        self.text_field_embedder = text_field_embedder
        self.target_field_embedder = target_field_embedder
        self.num_classes = self.vocab.get_vocab_size("labels")
        self.text_encoder = text_encoder
        self.target_encoder = target_encoder
        self.feedforward = feedforward

        if feedforward is not None:
            output_dim = self.feedforward.get_output_dim()
        elif target_encoder:
            target_out_dim = self.target_encoder.get_output_dim()
            text_out_dim = self.text_encoder.get_output_dim()
            output_dim = target_out_dim + text_out_dim
        else:
            output_dim = self.text_encoder.get_output_dim()
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
        self._variational_dropout = InputVariationalDropout(dropout)
        self._naive_dropout = Dropout(dropout)

        self.loss = torch.nn.CrossEntropyLoss()
        
        # Ensure that the dimensions of the text field embedder and text encoder
        # match
        check_dimensions_match(text_field_embedder.get_output_dim(), 
                               text_encoder.get_input_dim(),
                               "text field embedding dim", "text encoder input dim")
        # Ensure that the dimensions of the target or text field embedder and 
        # the target encoder match
        target_field_embedder_dim = text_field_embedder.get_output_dim()
        target_field_error = "text field embedding dim"
        if self.target_field_embedder:
            target_field_embedder_dim = target_field_embedder.get_output_dim()
            target_field_error = "target field embedding dim"
        if self.target_encoder:
            check_dimensions_match(target_field_embedder_dim, 
                                   target_encoder.get_input_dim(),
                                   target_field_error, "target encoder input dim")
        initializer(self)

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
        embedded_text = self.text_field_embedder(text)
        embedded_text = self._token_dropout(embedded_text)
        text_mask = util.get_text_field_mask(text)

        embedded_text = self._variational_dropout(embedded_text)
        encoded_text = self.text_encoder(embedded_text, text_mask)
        encoded_text = self._naive_dropout(encoded_text)

        if self.target_encoder:
            if self.target_field_embedder:
                embedded_target = self.target_field_embedder(target)
            else:
                embedded_target = self.text_field_embedder(target)
            embedded_target = self._token_dropout(embedded_target)
            embedded_target = self._variational_dropout(embedded_target)
            target_mask = util.get_text_field_mask(target)

            encoded_target = self.target_encoder(embedded_target, target_mask)
            encoded_target = self._naive_dropout(encoded_target)
            encoded_text = torch.cat([encoded_text, encoded_target], dim=-1)

        if self.feedforward:
            encoded_text = self.feedforward(encoded_text)
        logits = self.label_projection(encoded_text)
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