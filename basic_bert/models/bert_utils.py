# from .bert import CustomBertModel, BertConfig
from transformers import BertModel, LEDModel
# from pytorch_transformers import BertModel


# def build_bert(bert_config, pretrained=None):
#     if not pretrained:
#         bert = CustomBertModel(config=bert_config)
#     else:
#         bert = CustomBertModel(config=bert_config, bert=pretrained)
#     return bert 


def build_bert(model_name):
    try:
        bert = BertModel.from_pretrained(model_name)
        return bert 
    except:
        raise Exception("Invalid BERT model")

# def build_bert(model_name):
#     try:
#         bert = LEDModel.from_pretrained(model_name)
#         return bert 
#     except:
#         raise Exception("Invalid BERT model")
