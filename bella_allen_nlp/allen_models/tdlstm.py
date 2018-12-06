from typing import Dict, Optional

import numpy
from overrides import overrides
import torch
import torch.nn.functional as F

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary
from allennlp.modules import FeedForward, Seq2VecEncoder, TextFieldEmbedder
from allennlp.modules.seq2vec_encoders import BagOfEmbeddingsEncoder
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn import util
from allennlp.training.metrics import CategoricalAccuracy


@Model.register("tdlstm_classifier")
class TDLSTMClassifier(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 left_text_encoder: Seq2VecEncoder,
                 right_text_encoder: Seq2VecEncoder,
                 classifier_feedforward: FeedForward,
                 target_field_embedder: Optional[TextFieldEmbedder] = None,
                 target_encoder: Optional[Seq2VecEncoder] = None,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, regularizer)
        '''
        Having a target encoder converts this model into the original TCLSTM 
        model if the target encoder is the averaged Bag Of Embeddings Encoder.
        The encoded target will be concaenated to each word in the left and 
        right context before the right and left contexts are encoded.
        '''

        self.text_field_embedder = text_field_embedder
        self.target_field_embedder = target_field_embedder
        self.num_classes = self.vocab.get_vocab_size("labels")
        self.left_text_encoder = left_text_encoder
        self.right_text_encoder = right_text_encoder
        self.target_encoder = target_encoder
        self.classifier_feedforward = classifier_feedforward
        self.metrics = {
                "accuracy": CategoricalAccuracy()
        }
        self.loss = torch.nn.CrossEntropyLoss()
        initializer(self)

    def forward(self,
                left_text: Dict[str, torch.LongTensor],
                right_text: Dict[str, torch.LongTensor],
                target: Dict[str, torch.LongTensor],
                label: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
        '''
        The text and targets are Dictionaries as they are text fields they can 
        be represented many different ways e.g. just words or words and chars 
        etc therefore the dictionary represents these different ways e.g. 
        {'words': words_tensor_ids, 'chars': char_tensor_ids}
        '''
        left_embedded_text = self.text_field_embedder(left_text)
        right_embedded_text = self.text_field_embedder(right_text)
        left_text_mask = util.get_text_field_mask(left_text)
        right_text_mask = util.get_text_field_mask(right_text)

        if self.target_field_embedder:
            embedded_target = self.target_field_embedder(target)
        else:
            embedded_target = self.text_field_embedder(target)
        if self.target_encoder:
            target_text_mask = util.get_text_field_mask(target)
            target_encoded_text = self.target_encoder(embedded_target, 
                                                      target_text_mask)
            # Encoded target to be of dimension (batch, words, dim) currently
            # (batch, dim)
            target_encoded_text = target_encoded_text.unsqueeze(1)#.repeat((1,37,1))

            # Need to repeat the target word for each word in the left 
            # and right word.
            left_num_padded = left_embedded_text.shape[1]
            right_num_padded = right_embedded_text.shape[1]

            left_targets = target_encoded_text.repeat((1, left_num_padded, 1))
            right_targets = target_encoded_text.repeat((1, right_num_padded, 1))
            # Add the target to each word in the left and right contexts
            left_embedded_text = torch.cat((left_embedded_text, left_targets), -1)
            right_embedded_text = torch.cat((right_embedded_text, right_targets), -1)
        
        
        left_encoded_text = self.left_text_encoder(left_embedded_text, 
                                                   left_text_mask)
        right_encoded_text = self.right_text_encoder(right_embedded_text, 
                                                     right_text_mask)


        encoded_left_right = torch.cat([left_encoded_text, right_encoded_text], 
                                       dim=-1)
        logits = self.classifier_feedforward(encoded_left_right)
        class_probabilities = F.softmax(logits, dim=-1)

        output_dict = {"class_probabilities": class_probabilities}

        if label is not None:
            loss = self.loss(logits, label)
            for metric in self.metrics.values():
                metric(logits, label)
            output_dict["loss"] = loss

        return output_dict

    def decode(self, output_dict: Dict[str, torch.Tensor]
               ) -> Dict[str, torch.Tensor]:
        predictions = output_dict['class_probabilities'].cpu().data.numpy()
        argmax_indices = numpy.argmax(predictions, axis=-1)
        labels = [self.vocab.get_token_from_index(x, namespace="labels")
                  for x in argmax_indices]
        output_dict['label'] = labels
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {metric_name: metric.get_metric(reset) 
                for metric_name, metric in self.metrics.items()}
