from typing import Dict, Optional

from allennlp.common.checks import check_dimensions_match, ConfigurationError
from allennlp.data import Vocabulary
from allennlp.modules import FeedForward, Seq2SeqEncoder, TextFieldEmbedder, Seq2VecEncoder
from allennlp.modules import InputVariationalDropout, TimeDistributed
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn import util, Activation
from allennlp.training.metrics import CategoricalAccuracy, F1Measure
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.modules import Dropout, Dropout2d, Linear

from bella_allen_nlp.modules.cpt import CPT


@Model.register("transformation_target_classifier")
class TransformationTargetClassifier(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 text_encoder: Seq2SeqEncoder,
                 output_encoder: Seq2VecEncoder,
                 num_cpt_layers: Optional[int] = 2,
                 cpt_highway: bool = True,
                 target_encoder: Optional[Seq2SeqEncoder] = None,
                 feedforward: Optional[FeedForward] = None,
                 target_field_embedder: Optional[TextFieldEmbedder] = None,
                 attention_activation_function: Optional[str] = 'tanh',
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None,
                 share_text_target_encoder: bool = False,
                 word_dropout: float = 0.0,
                 variational_dropout: float = 0.0,
                 dropout: float = 0.0,
                 full_dropout: bool = False) -> None:
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

        if share_text_target_encoder and (target_encoder is not None):
            config_err = ("The target encoder will not be used when sharing. "
                          "Set the target_encoder to None (default)")
            raise ConfigurationError(config_err)
        elif (not share_text_target_encoder) and (target_encoder is None):
            config_err = ('As the target and text are not sharing the encoder '
                          'an encoder is required for the target text')
            raise ConfigurationError(config_err)

        self.text_field_embedder = text_field_embedder
        self.target_field_embedder = target_field_embedder
        self.num_classes = self.vocab.get_vocab_size("labels")
        self.text_encoder = text_encoder
        if share_text_target_encoder:
            target_encoder = text_encoder
        self.target_encoder = target_encoder
        self.output_encoder = output_encoder

        text_enc_out = text_encoder.get_output_dim()
        target_enc_out = target_encoder.get_output_dim()
        if full_dropout:
            self.cpt = TimeDistributed(CPT(num_cpt_layers, text_enc_out, 
                                           target_enc_out, cpt_highway,
                                           dropout=dropout))
        else:
            self.cpt = TimeDistributed(CPT(num_cpt_layers, text_enc_out, 
                                           target_enc_out, cpt_highway))
        
        self.feedforward = feedforward

        if feedforward is not None:
            output_dim = self.feedforward.get_output_dim()
        else:
            output_dim = self.output_encoder.get_output_dim()
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
        self.full_dropout = full_dropout

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
                label: torch.LongTensor = None,
                epoch_numbers: Optional[torch.Tensor] = None
                ) -> Dict[str, torch.Tensor]:
        '''
        The text and targets are Dictionaries as they are text fields they can 
        be represented many different ways e.g. just words or words and chars 
        etc therefore the dictionary represents these different ways e.g. 
        {'words': words_tensor_ids, 'chars': char_tensor_ids}
        '''
        # Embed and encode text as a sequence
        embedded_text = self.text_field_embedder(text)
        embedded_text = self._token_dropout(embedded_text)
        embedded_text = self._variational_dropout(embedded_text)
        text_mask = util.get_text_field_mask(text)
        
        encoded_text_seq = self.text_encoder(embedded_text, text_mask)
        if self.full_dropout:
            encoded_text_seq = self._variational_dropout(encoded_text_seq)

        # Embed target
        if self.target_field_embedder:
            embedded_target = self.target_field_embedder(target)
        else:
            embedded_target = self.text_field_embedder(target)
        embedded_target = self._token_dropout(embedded_target)
        embedded_target = self._variational_dropout(embedded_target)
        target_mask = util.get_text_field_mask(target)
        # Encode target
        encoded_target_seq = self.target_encoder(embedded_target, target_mask)
        if self.full_dropout:
            encoded_target_seq = self._variational_dropout(encoded_target_seq)
        # Need to add the dummy time element from the text for the time 
        # distributed CPT
        num_token_text = encoded_text_seq.shape[1]
        encoded_target_seq = encoded_target_seq.unsqueeze(1)
        encoded_target_seq = encoded_target_seq.expand(-1, num_token_text, -1, -1)
        target_mask = target_mask.unsqueeze(1)
        target_mask = target_mask.expand(-1, num_token_text, -1)
        cpt_text_seq = self.cpt(encoded_text_seq, encoded_target_seq, target_mask)
        # Output encoder
        output_vector = self.output_encoder(cpt_text_seq, text_mask)
        output_vector = self._naive_dropout(output_vector)

        if self.feedforward:
            output_vector = self.feedforward(output_vector)
        logits = self.label_projection(output_vector)
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