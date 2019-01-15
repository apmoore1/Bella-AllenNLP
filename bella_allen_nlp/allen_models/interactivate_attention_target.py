from typing import Dict, Optional

from allennlp.common.checks import check_dimensions_match
from allennlp.data import Vocabulary
from allennlp.modules.seq2vec_encoders import BagOfEmbeddingsEncoder
from allennlp.modules import FeedForward, Seq2SeqEncoder, TextFieldEmbedder
from allennlp.modules import InputVariationalDropout
from allennlp.modules.attention import BilinearAttention
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn import util, Activation
from allennlp.training.metrics import CategoricalAccuracy, F1Measure
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.modules import Dropout, Dropout2d, Linear


@Model.register("interactive_attention_target_classifier")
class InteractiveAttentionTargetClassifier(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 text_encoder: Seq2SeqEncoder,
                 target_encoder: Seq2SeqEncoder,
                 feedforward: Optional[FeedForward] = None,
                 target_field_embedder: Optional[TextFieldEmbedder] = None,
                 attention_activation_function: Optional[str] = 'tanh',
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
        :param attention_activation_function: The name of the activation 
                                              function applied after the 
                                              ``x^T W y + b`` calculation.
                                              Activation names can be found 
                                              `here <https://allenai.github.io/
                                              allennlp-docs/api/allennlp.nn.
                                              activations.html>`_. Default is 
                                              tanh.
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
        
        This attention target classifier is based on the model in `Exploiting  
        Document Knowledge for Aspect-level Sentiment Classification Ruidan 
        <https://aclanthology.info/papers/P18-2092/p18-2092>`_ where the 
        attention on the encoded context words are based on the encoded target 
        vector.
        '''
        super().__init__(vocab, regularizer)

        self.text_field_embedder = text_field_embedder
        self.target_field_embedder = target_field_embedder
        self.num_classes = self.vocab.get_vocab_size("labels")
        self.text_encoder = text_encoder
        self._text_averager = BagOfEmbeddingsEncoder(text_encoder.get_output_dim(), 
                                                     averaged=True)
        self.target_encoder = target_encoder
        self._target_averager = BagOfEmbeddingsEncoder(target_encoder.get_output_dim(), 
                                                       averaged=True)
        self.feedforward = feedforward
        attention_activation_function = Activation.by_name(f'{attention_activation_function}')()
        self.context_attention_layer = BilinearAttention(self.target_encoder.get_output_dim(),
                                                         self.text_encoder.get_output_dim(),
                                                         attention_activation_function,
                                                         normalize=True)
        self.target_attention_layer = BilinearAttention(self.text_encoder.get_output_dim(),
                                                        self.target_encoder.get_output_dim(),
                                                        attention_activation_function,
                                                        normalize=True)
        

        if feedforward is not None:
            output_dim = self.feedforward.get_output_dim()
        else:
            output_dim = (text_encoder.get_output_dim() + 
                          target_encoder.get_output_dim())
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
        # Embed and encode text as a sequence
        embedded_text = self.text_field_embedder(text)
        embedded_text = self._token_dropout(embedded_text)
        text_mask = util.get_text_field_mask(text)

        embedded_text = self._variational_dropout(embedded_text)
        encoded_text_seq = self.text_encoder(embedded_text, text_mask)
        encoded_text_seq = self._variational_dropout(encoded_text_seq)

        # Embed and encode target as a sequence
        if self.target_field_embedder:
            embedded_target = self.target_field_embedder(target)
        else:
            embedded_target = self.text_field_embedder(target)
        embedded_target = self._token_dropout(embedded_target)
        embedded_target = self._variational_dropout(embedded_target)
        target_mask = util.get_text_field_mask(target)

        encoded_target_seq = self.target_encoder(embedded_target, target_mask)
        encoded_target_seq = self._variational_dropout(encoded_target_seq)

        #
        # Attention
        #
        # context/text attention
        # Get average of the target hidden states as the query vector for the 
        # context attention
        avg_target_vec = self._target_averager(encoded_target_seq, target_mask)
        context_attention_weights = self.context_attention_layer(avg_target_vec,
                                                                 encoded_text_seq,
                                                                 text_mask)
        context_attention_weights = context_attention_weights.unsqueeze(-1)
        weighted_encoded_text_seq = encoded_text_seq * context_attention_weights
        weighted_encoded_text_vec = weighted_encoded_text_seq.sum(1)
        # target attention
        # Get average of the context hidden states as the query vector for the 
        # target attention
        avg_context_vec = self._text_averager(encoded_text_seq, text_mask)
        target_attention_weights = self.target_attention_layer(avg_context_vec,
                                                               encoded_target_seq,
                                                               target_mask)
        target_attention_weights = target_attention_weights.unsqueeze(-1)
        weighted_encoded_target_seq = encoded_target_seq * target_attention_weights
        weighted_encoded_target_vec = weighted_encoded_target_seq.sum(1)
        
        # Concatenate the two weighted text and target vectors
        weighted_text_target = torch.cat([weighted_encoded_text_vec, 
                                          weighted_encoded_target_vec], -1)
        weighted_text_target = self._naive_dropout(weighted_text_target)

        if self.feedforward:
            weighted_text_target = self.feedforward(weighted_text_target)
        # Putting it through a tanh first and then a softmax
        logits = self.label_projection(weighted_text_target)
        logits = torch.tanh(logits)
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