"""
A feed-forward neural network.
"""
import torch
from torch.nn import Linear, Dropout, Hardtanh

from allennlp.modules.attention import DotProductAttention


class CPT(torch.nn.Module):
    '''
    Context-Preserving Transformation (CPT) module from the `Transformation 
    Networks for Target-Oriented Sentiment Classification paper 
    <https://aclanthology.info/papers/P18-1087/p18-1087>`_.
    '''
    def __init__(self, num_layers: int, text_encoder_out_dim: int, 
                 target_encoder_out_dim: int, highway: bool = True,
                 dropout: float = 0.0) -> None:
        '''
        :param num_layers: Number of times to perform the CPT layer
        :param text_encoder_out_dim: The output dimension of the text encoder
        :param target_encoder_out_dim: The output dimension of the target 
                                       encoder
        '''
        super().__init__()
        target_text_enc_out = target_encoder_out_dim + text_encoder_out_dim
        self.cpt_feedforward = Linear(target_text_enc_out, text_encoder_out_dim)
        self.attention = DotProductAttention(normalize=True)
        self.num_layers = num_layers
        self._highway = highway
        self._activation = Hardtanh()
        self._naive_dropout = Dropout(dropout)
        self._output_dim = text_encoder_out_dim

    def forward(self, text_vec: torch.Tensor, target_seq: torch.Tensor, 
                target_mask: torch.Tensor) -> torch.Tensor:
        '''
        :param text_vec: The vector representing one token in the context text.
        :param target_seq: The vectors representing all the tokens in the 
                           target.
        :param target_mask: The mask for the target sequence.
        '''
        for layer in range(self.num_layers):
            # Find which words in the target sequence are important for each 
            # text token
            attention_weights = self.attention(text_vec, target_seq, target_mask)
            weighted_targets_seq = torch.mul(target_seq, 
                                            attention_weights.unsqueeze(-1))
            weighted_targets_vec = weighted_targets_seq.sum(1)
            # Similarity between the text and the weighted target
            text_targets = torch.cat((text_vec, weighted_targets_vec), -1)
            infused_text_target = self.cpt_feedforward(text_targets)
            infused_text_target = self._activation(infused_text_target)
            infused_text_target = self._naive_dropout(infused_text_target)
            # Context preserving
            if self._highway:
                infused_text_target = infused_text_target + text_vec
            text_vec = infused_text_target
        return infused_text_target

    def get_output_dim(self):
        return self._output_dim