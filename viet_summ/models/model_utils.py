import torch 
from .model_map import MODEL_MAP
import torch.nn as nn

def build_model(config, device, checkpoint=None):
    model_class = MODEL_MAP[config.model_name]
    model = model_class(config)
    
    if checkpoint:
        model.load_state_dict(checkpoint['model'], strict=True)
    model.to(device)
    return model