from models.phobert import *
from .basic_models import *
from .led_utils import *

MODEL_MAP = {
    'basic_ext': BasicTransformerSentenceClassification, 
    'basic_abs': BasicTransformerSentenceGeneration,
    'led_abs': LEDBasicSentenceGenerationModel, 
    'led_ext' : LEDBasicSentenceClassificationModel,

    'vi_basic_ext': BasicViTransformerSentenceClassification,
    'vi_basic_abs': BasicViTransformerSentenceGeneration,
}

