from distutils.command.build import build
import torch 
import torch.nn as nn 
from .modules.transformer_base import (BasicTransformerModel, 
                                        BasicTransformerEncoder, 
                                        BasicTransformerEncoderBlock, 
                                        PositionwiseFeedForward,)
from .bert import PositionalEncoding
from .bert_utils import build_bert


def get_generator(vocab_size, dec_hidden_size):
    gen_func = nn.LogSoftmax(dim=-1)
    generator = nn.Sequential(
        nn.Linear(dec_hidden_size, vocab_size),
        gen_func
    )
    generator

class BasicTransformerSentenceGeneration(nn.Module):
    def __init__(self, args, **kwargs):
        super(BasicTransformerSentenceGeneration, self).__init__()
        
        self.bert = build_bert(args.bert_config) 
        if args.freeze_bert:
            self.bert.eval()
        else: 
            self.bert.train()

        self.transformer_model = BasicTransformerModel(
            d_model=args.d_model, 
            num_heads=args.num_heads, 
            num_encoder_blocks=args.num_encoder_blocks,
            num_decoder_blocks=args.num_decoder_blocks, 
            dim_ff=args.dim_ff,
            dropout=args.dropout,
            batch_first=args.batch_first, 
            norm_first=args.norm_first, 
            layer_norm_eps=args.layer_norm_eps
        )

        self.generator = get_generator(args.vocab_size, args.d_model)
        self.vocab_size = args.vocab_size
        self.args = args

    def forward(self, src, tgt, segs, docs, mask_src, mask_tgt):
        top_vec, _ = self.bert(src, attention_mask=mask_src, 
                                token_type_ids=segs, doc_type_ids=docs)
        
        output = self.transformer_model(top_vec, tgt, mask_src, mask_tgt, )
        return output

class BasicTransformerSentenceClassification(nn.Module):
    def __init__(self, args, **kwargs):
        super(BasicTransformerSentenceClassification, self).__init__()

        self.bert = build_bert(args.bert_model) 
        if args.freeze_bert:
            self.bert.eval()
        else: 
            self.bert.train()
        self.bert.embeddings.register_buffer("position_ids", torch.arange(args.max_position_embeddings).expand((1, -1)))
        # self.bert.register_buffer(
        #         "token_type_ids",
        #         torch.zeros(self.bert.position_ids.size(), dtype=torch.long),
        #         persistent=False,
        #     )
            
        if(args.max_position_embeddings>512):
            my_pos_embeddings = nn.Embedding(args.max_position_embeddings, self.bert.config.hidden_size)
            my_pos_embeddings.weight.data[:512] = self.bert.embeddings.position_embeddings.weight.data
            my_pos_embeddings.weight.data[512:] = self.bert.embeddings.position_embeddings.weight.data[-1][None,:].repeat(args.max_position_embeddings-512,1)
            self.bert.embeddings.position_embeddings = my_pos_embeddings

        self.pos_emb = PositionalEncoding(args.d_model, args.max_position_embeddings, args.dropout)
        self.doc_type_embeddings = nn.Embedding(args.type_doc_size, args.d_model)
        self.norm = nn.LayerNorm(args.d_model, eps=1e-6)

        encoder_block = BasicTransformerEncoderBlock(args.d_model, args.num_heads, args.d_ff, 
                                                    args.dropout, args.norm_first)
        self.encoder = BasicTransformerEncoder(encoder_block, args.num_encoder_blocks, 
                            nn.LayerNorm(args.d_model, eps=args.layer_norm_eps))

        for p in self.encoder.parameters():
            if p.dim()>1:
                nn.init.xavier_uniform_(p)

        self.wo = nn.Linear(args.d_model, 1, bias=True)
        nn.init.xavier_uniform_(self.wo.weight)
        self.sigmoid = nn.Sigmoid()

    def forward(self, src, segs, clss, mask_src, mask_cls):
        # pytorch_transformers 
        # top_vec, _ = self.bert(input_ids=src,
        #                     attention_mask=mask_src,
        #                     token_type_ids=segs, )

        # huggingface bert
        top_vec, _ = self.bert(input_ids=src,
                            attention_mask=mask_src,
                            token_type_ids=segs, return_dict=False)

        # doc_embeddings = self.doc_type_embeddings(docs) 
        # top_vec = top_vec + doc_embeddings
        # top_vec = self.norm(top_vec)
        

        sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss]
        sents_vec = sents_vec * mask_cls[:, :, None].float()   

        # with cosine positional
        pos_emb = self.pos_emb.pe[:, :sents_vec.size(1)]
        sents_vec = sents_vec * mask_cls[:, :, None].float()
        sents_vec = sents_vec + pos_emb

        x = self.encoder(sents_vec, mask_cls)
        sent_scores = self.sigmoid(self.wo(x))
        sent_scores = sent_scores.squeeze(-1) * mask_cls.float()
        return sent_scores, mask_cls