from dataclasses import dataclass, asdict, fields
from models.bert import BertConfig

@dataclass(init=True, repr=True)
class ModelConfig:
    freeze_bert: bool
    d_model: int
    num_heads: int
    dropout: float 
    norm_first: bool
    num_encoder_blocks: int
    layer_norm_eps: float
    d_ff: int
    bert_config: BertConfig

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if k == 'bert_config':
                self.bert_config = BertConfig(**v)
            else:
                setattr(self, k, v)



