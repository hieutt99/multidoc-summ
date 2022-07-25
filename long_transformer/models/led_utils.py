import copy
from glob import glob
from models.basic_models import get_generator
from models.modules.transformer_base import TransformerDecoder
import torch.nn as nn 
import torch 
from transformers import LEDModel
from .bert import PositionalEncoding

class LEDClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
        self,
        input_dim: int,
        inner_dim: int,
        num_classes: int,
        pooler_dropout: float,
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        # hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states

def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    # if pad_token_id is None:
    #     raise ValueError("config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids

class LEDBasicSentenceClassificationModel(nn.Module):
    def __init__(self, args, tokenizer, **kwargs):
        super(LEDBasicSentenceClassificationModel, self).__init__(**kwargs)
        self.pad_token_id = tokenizer.pad_token_id
        self.decoder_start_token_id = tokenizer.bos_token_id

        self.model_name = args.model_name

        # self.bert = LEDModel.from_pretrained(args.bert_model)

        led_model = LEDModel.from_pretrained(args.bert_model).to('cpu')
        led_model.resize_token_embeddings(len(tokenizer))
        self.bert = copy.deepcopy(led_model.get_encoder())
        self.bert.layers = self.bert.layers[:3]
        
        self.decoder = led_model.get_decoder()
        self.decoder.layers = self.decoder.layers[:3]
        self.decoder.embed_positions = copy.deepcopy(self.bert.embed_positions)

        self.bert.train()
        self.decoder.train()

        # self.classifer = LEDClassificationHead(args.d_model, args.d_model, 1, args.dropout)
        self.classifer = LEDClassificationHead(args.d_model, 3072, 1, args.dropout)

        self.bert._init_weights(self.classifer.dense)
        self.bert._init_weights(self.classifer.out_proj)
        self.sigmoid = nn.Sigmoid()


    def forward(self, src, segs, glob_mask, clss, mask_src, mask_cls):
        
        led_outputs = self.bert(input_ids=src,
                            attention_mask=mask_src,
                            global_attention_mask=glob_mask,
                            return_dict=False)
        top_vec = self.decoder(
            input_ids=shift_tokens_right(src, self.pad_token_id, self.decoder_start_token_id),
            encoder_hidden_states=led_outputs[0], return_dict=False)[0]

        sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss]
        sents_vec = sents_vec * mask_cls[:, :, None].float() 

        sent_scores = self.classifer(sents_vec)
        if not self.training:
            sent_scores = self.sigmoid(sent_scores)
        sent_scores = sent_scores.squeeze(-1) * mask_cls.float()
        return sent_scores, mask_cls

class LEDBasicSentenceGenerationModel(nn.Module):
    def __init__(self, args, tokenizer, **kwargs):
        super(LEDBasicSentenceGenerationModel, self).__init__()

        self.model_name = args.model_name
        
        led_model = LEDModel.from_pretrained(args.bert_model).to('cpu')
        led_model.resize_token_embeddings(len(tokenizer))
        self.bert = led_model.get_encoder()

        if args.freeze_bert:
            self.bert.eval()
        else: 
            self.bert.train()
            
        tgt_embeddings = nn.Embedding(len(tokenizer), args.d_model, padding_idx=tokenizer.pad_token_id)
        # tgt_embeddings.weight = copy.deepcopy(self.bert.shared.weight)
        # tgt_embeddings.weight = copy.deepcopy(self.bert.decoder.embed_tokens.weight)

        tgt_embeddings.weight = copy.deepcopy(self.bert.embed_tokens.weight)

        # tgt_embeddings.weight = self.bert.embed_tokens.weight

        self.decoder = TransformerDecoder(
            args.dec_layers,
            args.dec_hidden_size, heads=args.dec_heads,
            d_ff=args.dec_ff_size, dropout=args.dec_dropout, embeddings=tgt_embeddings)

        self.generator = get_generator(len(tokenizer), args.d_model)
        self.generator[0].weight = self.decoder.embeddings.weight
        self.vocab_size = len(tokenizer)
        self.args = args

        for module in self.decoder.modules():
            if isinstance(module, (nn.Linear)):
                module.weight.data.normal_(mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        for p in self.generator.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                p.data.zero_()

    def forward(self, src, tgt, segs, mask_src, clss, mask_cls, glob_mask):
        
        led_outputs = self.bert(input_ids=src,
                            attention_mask=mask_src,
                            # token_type_ids=segs, 
                            # global_attention_mask=glob_mask,
                            return_dict=False)
        top_vec = led_outputs[0]

        dec_state = self.decoder.init_decoder_state(src, top_vec)
        decoder_outputs, state = self.decoder(tgt[:, :-1], top_vec, dec_state)
        return decoder_outputs, None
