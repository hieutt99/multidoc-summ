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
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states

class LEDBasicSentenceClassificationModel(nn.Module):
    def __init__(self, args, **kwargs):
        super(LEDBasicSentenceClassificationModel, self).__init__(**kwargs)

        self.model_name = args.model_name

        self.bert = LEDModel.from_pretrained(args.bert_model)
        self.bert.train()
        
        self.pos_emb = PositionalEncoding(args.d_model, args.max_position_embeddings, args.dropout)

        self.classifer = LEDClassificationHead(args.d_model, args.d_model, 1, args.dropout)

        self.bert._init_weights(self.classifer.dense)
        self.bert._init_weights(self.classifer.out_proj)
        # self.classifier = nn.Linear(args.d_model, 1)
        # self.dropout = nn.Dropout(args.dropout)

        self.sigmoid = nn.Sigmoid()


    def forward(self, src, segs, glob_mask, clss, mask_src, mask_cls):
        
        led_outputs = self.bert(input_ids=src,
                            attention_mask=mask_src,
                            # token_type_ids=segs, 
                            global_attention_mask=glob_mask,
                            return_dict=False)
        top_vec = led_outputs[0]

        sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss]
        sents_vec = sents_vec * mask_cls[:, :, None].float() 

        # # with cosine positional
        # pos_emb = self.pos_emb.pe[:, :sents_vec.size(1)]
        # sents_vec = sents_vec * mask_cls[:, :, None].float()
        # sents_vec = sents_vec + pos_emb

        sent_scores = self.classifer(sents_vec)
        sent_scores = self.sigmoid(sent_scores)
        sent_scores = sent_scores.squeeze(-1) * mask_cls.float()
        return sent_scores, mask_cls

class LEDBasicSentenceGenerationModel(nn.Module):
    def __init__(self, args, tokenizer, **kwargs):
        super(LEDBasicSentenceGenerationModel, self).__init__()

        self.model_name = args.model_name
        
        self.bert = LEDModel.from_pretrained(args.bert_model)
        self.bert.resize_token_embeddings(len(tokenizer))
        if args.freeze_bert:
            self.bert.eval()
        else: 
            self.bert.train()
        # self.bert.embeddings.register_buffer("position_ids", torch.arange(args.max_position_embeddings).expand((1, -1)))
            
        # if(args.max_position_embeddings>512):
        #     my_pos_embeddings = nn.Embedding(args.max_position_embeddings, self.bert.config.hidden_size)
        #     my_pos_embeddings.weight.data[:512] = self.bert.embeddings.position_embeddings.weight.data
        #     my_pos_embeddings.weight.data[512:] = self.bert.embeddings.position_embeddings.weight.data[-1][None,:].repeat(args.max_position_embeddings-512,1)
        #     self.bert.embeddings.position_embeddings = my_pos_embeddings

        # self.pos_emb = PositionalEncoding(args.d_model, args.max_position_embeddings, args.dropout)
        # self.doc_type_embeddings = nn.Embedding(args.type_doc_size, args.d_model)

        tgt_embeddings = nn.Embedding(len(tokenizer), args.d_model, padding_idx=tokenizer.pad_token_id)
        # tgt_embeddings.weight = copy.deepcopy(self.bert.shared.weight)
        tgt_embeddings.weight = copy.deepcopy(self.bert.decoder.embed_tokens.weight)

        self.decoder = TransformerDecoder(
            args.dec_layers,
            args.dec_hidden_size, heads=args.dec_heads,
            d_ff=args.dec_ff_size, dropout=args.dec_dropout, embeddings=tgt_embeddings)

        self.generator = get_generator(len(tokenizer), args.d_model)
        self.generator[0].weight = self.decoder.embeddings.weight
        self.vocab_size = len(tokenizer)
        self.args = args

        for module in self.decoder.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
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
                            global_attention_mask=glob_mask,
                            return_dict=False)
        top_vec = led_outputs[0]

        dec_state = self.decoder.init_decoder_state(src, top_vec)
        decoder_outputs, state = self.decoder(tgt[:, :-1], top_vec, dec_state)
        return decoder_outputs, None
