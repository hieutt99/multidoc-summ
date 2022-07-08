from .basic_models import *
from .led_utils import *

MODEL_MAP = {
    'basic_ext': BasicTransformerSentenceClassification, 
    'basic_abs': BasicTransformerSentenceGeneration,
    'led_abs': LEDBasicSentenceGenerationModel, 
    'led_ext' : LEDBasicSentenceClassificationModel,
}

