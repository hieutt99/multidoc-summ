from dataclasses import dataclass, asdict, fields
from re import S
from typing import Optional, Type, List
from models.bert import BertConfig
from utils.utils import load_config_yaml
import os 

@dataclass(init=True, repr=True)
class ModelConfig:
    model_name: str
    vocab_size: int
    freeze_bert: bool
    d_model: int
    num_heads: int
    dropout: float 
    norm_first: bool
    num_encoder_blocks: int
    num_decoder_blocks: int
    layer_norm_eps: float
    d_ff: int

    dec_layers: int
    dec_hidden_size: int
    dec_heads: int
    dec_ff_size: int
    dec_dropout: float
    # bert_config: BertConfig
    max_position_embeddings: int
    bert_model: str = 'bert-base-cased'
    type_doc_size: int = 2
    padding_idx: int = 0

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if k == 'bert_config':
                self.bert_config = BertConfig(**v) if type(v) == dict else v
            else:
                setattr(self, k, v)


@dataclass(init=True, repr=True)
class RunConfig:
    task: str
    mode: str

    bert_data_path: str
    model_path: str
    result_path: str
    temp_dir: str

    model_config_path: str
    model_config: ModelConfig
    checkpoint: str
    
    train_from: str
    test_from: str
    log_file: str

    block_trigram: bool
    report_rouge: bool

    batch_size: int
    test_batch_size: int

    max_pos: int 
    use_interval: bool

    log_file: str 
    seed: int 
    visible_gpus: str

    param_init: float
    param_init_glorot: bool

    optim: str
    lr: float
    beta1: float
    beta2: float
    warmup_steps: int
    warmup_steps_bert: int
    warmup_steps_dec: int
    max_grad_norm: int

    save_checkpoint_steps: int
    accum_count: int
    report_every: int
    train_steps: int
    recall_eval: bool

    label_smoothing: float
    generator_shard_size: int
    alpha: float
    beam_size: int
    min_length: int
    max_length: int
    max_tgt_len: int 
    ext_predict_nsents: int

    visible_gpus: str
    gpu_ranks: ...
    world_size: int
    seed: int

    lr_bert: float
    lr_dec: float

    test_all: bool = False

    load_from_extractive: str = ""
    sep_optim: bool = False
    test_start_from: int = -1

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if k != 'model_config':
                setattr(self, k, v)
        if os.path.exists(self.model_config_path):
            self.model_config = ModelConfig(**load_config_yaml(self.model_config_path))
        else:
            raise Exception("model config file invalid") 
        if not self.train_from:
            self.train_from = ''
        if not self.test_from:
            self.test_from = ''

def load_config(run_config_path):
    if os.path.exists(run_config_path):
        run_config = RunConfig(**load_config_yaml(run_config_path))
    else:
        raise Exception("run config file invalid")
    return run_config 
