from .basic_models import *
from .phobert import *

MODEL_MAP = {
    'basic_ext': BasicTransformerSentenceClassification, 
    'basic_abs': BasicTransformerSentenceGeneration,

    'vi_basic_ext': BasicViTransformerSentenceClassification,
}

