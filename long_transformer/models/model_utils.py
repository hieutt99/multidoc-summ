import torch 
from .model_map import MODEL_MAP

def build_model(config, device, checkpoint=None, tokenizer=None):
    model_class = MODEL_MAP[config.model_name]
    model = model_class(config)
    
    if tokenizer:
        model.resize_token_embeddings(len(tokenizer))

    if checkpoint:
        model.load_state_dict(checkpoint['model'], strict=True)
    model.to(device)
    return model
