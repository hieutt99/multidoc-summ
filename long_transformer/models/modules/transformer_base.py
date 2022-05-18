from json import decoder
import torch 
import torch.nn as nn 
from .attention import *
from .utils import _get_clones
from .attention import MultiHeadedAttention

class PositionwiseFeedForward(nn.Module):
    """ A two-layer Feed-Forward-Network with residual layer norm.

    Args:
        d_model (int): the size of input for the first-layer of the FFN.
        d_ff (int): the hidden layer size of the second-layer
            of the FNN.
        dropout (float): dropout probability in :math:`[0, 1)`.


    Input:
        x (batch_size, x, d_model)
    Output 
        output (batch_size, x, d_model)
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.activation = nn.GELU()
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        inter = self.dropout_1(self.activation(self.w_1(x)))
        output = self.dropout_2(self.w_2(inter))
        return output + x

class BasicTransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, d_ff, dropout, norm_first=False):
        super(BasicTransformerEncoderBlock, self).__init__()

        self.self_attention = MultiHeadedAttention(embed_dim, num_heads, dropout)
        """
        MultiheadAttention 
        input : query, key, value, key_padding_mask, need_weights(default True to return weights), 
                attn_mask, average_attn_weights
        output : attn_output, attn_output_weights


        embed_dim is the dimension of query, key, value 
        num_heads is the number of parallel heads 

        embed_dim will be split across num_heads which means embed_dim % num_heads == 0

        forward(query, key, value)
        """
        self.pff = PositionwiseFeedForward(embed_dim, d_ff, dropout)
        self.layer_norm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.dropout1 = nn.Dropout(dropout)
        self.norm_first = norm_first

    def forward(self, src, mask: Optional[torch.Tensor]=None, 
        src_key_padding_mask: Optional[torch.Tensor]=None):
        x = src
        if self.norm_first:
            x = x + self.self_attention_block(self.layer_norm1(x), mask, src_key_padding_mask)
            x = x + self.pff(self.layer_norm2(x))
        else:
            mask = mask.unsqueeze(1)
            x = self.layer_norm1(x + self.self_attention_block(x, mask, src_key_padding_mask))
            x = self.layer_norm2(x + self.pff(x))

        return x

    def self_attention_block(self, x, attn_mask, key_padding_mask):
        x = self.self_attention(x, x, x, mask=attn_mask)
        return self.dropout1(x)

class BasicTransformerDecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff=2048, dropout=0.1, norm_first=False):
        super(BasicTransformerDecoderBlock, self).__init__()
        
        self.self_attn = MultiHeadedAttention(d_model, num_heads, dropout=dropout)
        self.multihead_attn = MultiHeadedAttention(d_model, num_heads, dropout=dropout)

        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm_1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm_2 = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm_3 = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm_first=norm_first


    def forward(self, tgt, memory, tgt_mask=None,
                            memory_mask=None, 
                            tgt_key_padding_mask=None,
                            memory_key_padding_mask=None):
        x = tgt
        if self.norm_first:
            x = x + self.self_attn(self.layer_norm_1(x), tgt_mask, tgt_key_padding_mask)
            x = x + self.multihead_attention_block(self.layer_norm_2(x), memory, memory_mask, 
                                                    memory_key_padding_mask,)
            x = x + self.feed_forward(self.layer_norm_3(x))
        else:
            x = self.layer_norm_1(x + self.self_attention_block(x, tgt_mask, tgt_key_padding_mask))
            x = self.layer_norm_2(x + self.multihead_attention_block(x, memory, memory_mask, 
                                                                    memory_key_padding_mask))
            x = self.layer_norm_3(self.feed_forward(x))
        return x

    def self_attention_block(self, x, attn_mask, key_padding_mask):
        x = self.self_attention(x, x, x, mask=attn_mask)
        return self.dropout1(x)

    def multihead_attention_block(self, x, mem, attn_mask, key_padding_mask):
        x = self.multihead_attn(x, mem, mem, 
                                attn_mask=attn_mask, 
                                key_padding_mask=key_padding_mask, 
                                need_weights=False)[0]
        return self.dropout2(x)

class BasicTransformerEncoder(nn.Module):
    def __init__(self, encoder_block, num_blocks, norm=None):
        super(BasicTransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_block, num_blocks)
        self.num_blocks = num_blocks
        self.norm = norm 

    def forward(self, src, mask=None, src_key_padding_mask=None):
        output = src 
        
        for layer in self.layers:
            output = layer(output, mask=mask, 
                src_key_padding_mask=src_key_padding_mask)
        if self.norm is not None:
            output = self.norm(output)

        return output

class BasicTransformerDecoder(nn.Module):
    def __init__(self, decoder_block, num_blocks, norm=None):
        self.layers = _get_clones(decoder_block, num_blocks)
        self.num_blocks = num_blocks
        self.norm = norm 

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, 
        tgt_key_padding_mask=None, memory_key_padding_mask=None):
        output = tgt

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                            memory_mask=memory_mask, 
                            tgt_key_padding_mask=tgt_key_padding_mask,
                            memory_key_padding_mask=memory_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output

class BasicTransformerModel(nn.Module):
    def __init__(self, d_model=512, num_heads=8, num_encoder_blocks=6, 
                num_decoder_blocks=6, dim_ff=2048, dropout=0.1, batch_first=True, 
                norm_first=False, layer_norm_eps=1e-6):
        super(BasicTransformerModel, self).__init__()
        encoder_block = BasicTransformerEncoderBlock(d_model, num_heads, dropout, norm_first)
        self.encoder = BasicTransformerEncoder(encoder_block, num_encoder_blocks, 
                            nn.LayerNorm(d_model, eps=layer_norm_eps))
        decoder_block = BasicTransformerDecoderBlock(d_model, num_heads, dim_ff, dropout, norm_first)
        self.decoder = BasicTransformerDecoder(decoder_block, num_decoder_blocks,
                            nn.LayerNorm(d_model, eps=layer_norm_eps))

        self._reset_parameters()
        self.d_model = d_model 
        self.num_heads = num_heads 
        self.batch_first = batch_first

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, 
                memory_mask=None, src_key_padding_mask=None, 
                tgt_key_padding_mask=None, 
                memory_key_padding_mask=None):

        is_batched = src.dim() == 3
        if not self.batch_first and src.size(1) != tgt.size(1) and is_batched:
            raise RuntimeError("the batch number of src and tgt must be equal")
        elif self.batch_first and src.size(0) != tgt.size(0) and is_batched:
            raise RuntimeError("the batch number of src and tgt must be equal")
        if src.size(-1) != self.d_model or tgt.size(-1) != self.d_model:
            raise RuntimeError("the feature number of src and tgt must be equal to d_model")

        memory = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask, 
                                tgt_key_padding_mask=tgt_key_padding_mask, 
                                memory_key_padding_mask=memory_key_padding_mask)
        return output

    @staticmethod
    def generate_square_subsequent_mask(size:int):
        return torch.triu(torch.full((size, size), float('-inf')), diagonal=1)

    def _reset_parameters(self, ):
        for p in self.parameters():
            if p.dim()>1:
                nn.init.xavier_uniform_(p)

