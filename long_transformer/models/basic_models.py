from distutils.command.build import build
import torch 
import torch.nn as nn 
from .modules.transformer_base import (BasicTransformerModel, 
                                        BasicTransformerEncoder, 
                                        BasicTransformerEncoderBlock, 
                                        PositionwiseFeedForward,)
from bert import PositionalEncoding
from .bert_utils import build_bert

class BasicTransformerSentenceGeneration(nn.Module):
    def __init__(self, args, **kwargs):
        super(BasicTransformerSentenceGeneration, self).__init__()
        
        self.bert = build_bert(args.bert_config) 
        if args.freeze_bert:
            self.bert.eval()

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


        self.args = args

    def forward(self, src, tgt, segs, docs, mask_src, mask_tgt):
        top_vec, _ = self.bert(src, attention_mask=mask_src, 
                                token_type_ids=segs, doc_type_ids=docs)
        
        return 

class BasicTransformerSentenceClassification(nn.Module):
    def __init__(self, args, **kwargs):
        super(BasicTransformerSentenceClassification, self).__init__()

        self.bert = build_bert(bert_config=args.bert_config) 
        if args.freeze_bert:
            self.bert.eval()

        self.pos_emb = PositionalEncoding(args.d_model, args.max_position_embeddings, args.dropout)

        encoder_block = BasicTransformerEncoderBlock(args.d_model, args.num_heads, args.d_ff, 
                                                    args.dropout, args.norm_first)
        self.encoder = BasicTransformerEncoder(encoder_block, args.num_encoder_blocks, 
                            nn.LayerNorm(args.d_model, eps=args.layer_norm_eps))

        for p in self.encoder.parameters():
            if p.dim()>1:
                nn.init.xavier_uniform_(p)

        self.wo = nn.Linear(args.d_model, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, src, segs, docs, clss, mask_src, mask_cls):
        top_vec, _ = self.bert(input_ids=src,
                            attention_mask=mask_src,
                            token_type_ids=segs, 
                            doc_type_ids=docs)

        

        sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss]
        sents_vec = sents_vec * mask_cls[:, :, None].float()


        pos_emb = self.pos_emb.pe[:, :sents_vec.size(1)]
        sents_vec = sents_vec * mask_cls[:, :, None].float()
        sents_vec = sents_vec + pos_emb

        x = self.encoder(sents_vec, mask_cls).squeeze(-1)
        sent_scores = self.sigmoid(self.wo(x))
        sent_scores = sent_scores.squeeze(-1) * mask_cls.float()
        return sent_scores, mask_cls