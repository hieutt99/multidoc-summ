from .bert import CustomBertModel, BertConfig



def build_bert(bert_config, pretrained=None):
    if not pretrained:
        bert = CustomBertModel(config=bert_config)
    else:
        bert = CustomBertModel(config=bert_config, bert=pretrained)
    return bert 

