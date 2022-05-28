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

class LEDBasicSentenceClassificationModel(nn.Module):
    def __init__(self, args, **kwargs):
        super(LEDBasicSentenceClassificationModel, self).__init__(**kwargs)

        self.led_model = LEDModel.from_pretrained(args.bert_model)
        
        self.pos_emb = PositionalEncoding(args.d_model, args.max_position_embeddings, args.dropout)

        self.classifer = LEDClassificationHead(args.d_model, args.d_model, 1, args.dropout)

        self.led_model._init_weights(self.classifer.dense)
        self.led_model._init_weights(self.classifer.out_proj)
        # self.classifier = nn.Linear(args.d_model, 1)
        # self.dropout = nn.Dropout(args.dropout)

        self.sigmoid = nn.Sigmoid()


    def forward(self, src, segs, docs, clss, mask_src, mask_cls):
        led_outputs = self.led_model(input_ids=src,
                            attention_mask=mask_src,
                            # token_type_ids=segs, 
                            global_attention_mask=torch.zeros_like(mask_src),
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