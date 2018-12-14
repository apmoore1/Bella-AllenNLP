from typing import Dict, Optional

from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary
from allennlp.modules import FeedForward, Seq2VecEncoder, TextFieldEmbedder
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn import util
from allennlp.training.metrics import CategoricalAccuracy, F1Measure
import numpy as np
import torch
import torch.nn.functional as F

'''
Seq2VecEncoder is an abstract method that maps tensors of shape 
(batch_size, sequence_length, embedding_dim) to tensors of shape 
(batch_size, embedding_dim)
'''


@Model.register("target_classifier")
class TargetClassifier(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 text_encoder: Seq2VecEncoder,
                 classifier_feedforward: FeedForward,
                 target_field_embedder: Optional[TextFieldEmbedder] = None,
                 target_encoder: Optional[Seq2VecEncoder] = None,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, regularizer)

        self.text_field_embedder = text_field_embedder
        self.target_field_embedder = target_field_embedder
        self.num_classes = self.vocab.get_vocab_size("labels")
        self.text_encoder = text_encoder
        self.target_encoder = target_encoder
        self.classifier_feedforward = classifier_feedforward
        self.metrics = {
                "accuracy": CategoricalAccuracy()
        }
        self.f1_metrics = {}

        # Ensure that when the target encoder is used the input dim to the 
        # feedforward layer is expecting the right dimension. Easy to forget to 
        # add the text encoder out and the target encoder out
        if self.target_encoder is not None:
            feed_input_dim = self.classifier_feedforward.get_input_dim()
            target_out = self.target_encoder.get_output_dim()
            text_out = self.text_encoder.get_output_dim()

            config_err_msg = ("The input dim to the feedforward layer " 
                              f"{feed_input_dim} has to be the sum of the "
                              f"output of the target encoder {target_out} + "
                              f"the text encoder {text_out}")
            if feed_input_dim != (target_out + text_out):
                raise ConfigurationError(config_err_msg)

            if self.target_field_embedder is not None:
                target_embed_out = self.target_field_embedder.get_output_dim()
                target_in = self.target_encoder.get_input_dim()
                config_embed_err_msg = ("The Target field embedder should have"
                                        " the same output size "
                                        f"{target_embed_out} as the input to "
                                        f"the target encoder {target_in}")
                if target_embed_out != target_in:
                    raise ConfigurationError(config_embed_err_msg)
        # F1 Scores
        label_index_name = self.vocab.get_index_to_token_vocabulary('labels')
        for label_index, label_name in label_index_name.items():
            label_name = f'F1_{label_name.capitalize()}'
            self.f1_metrics[label_name] = F1Measure(label_index)
    
        self.loss = torch.nn.CrossEntropyLoss()
        initializer(self)

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
        text_mask = util.get_text_field_mask(text)
        encoded_text = self.text_encoder(embedded_text, text_mask)

        if self.target_encoder:
            if self.target_field_embedder:
                embedded_target = self.target_field_embedder(target)
            else:
                embedded_target = self.text_field_embedder(target)
            target_mask = util.get_text_field_mask(target)
            encoded_target = self.target_encoder(embedded_target, target_mask)
            encoded_text = torch.cat([encoded_text, encoded_target], dim=-1)

        logits = self.classifier_feedforward(encoded_text)
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