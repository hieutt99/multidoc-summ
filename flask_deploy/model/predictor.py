


def build_predictor():
    pass 


class Predictor():
    def __init__(self, ext_model, abs_model, ext_tokenizer, abs_tokenizer):
        self.ext_model = ext_model
        self.abs_model = abs_model

        self.ext_tokenizer = ext_tokenizer 
        self.abs_tokenizer = abs_tokenizer

    def preprocess(self, inp, mode):

        assert type(inp) == list

        if mode == 'ext':
            pass 
        elif mode == 'abs':
            pass 
        else:
            print("ERROR preprocess")
            return None

    def sent_classify(self, ):
        pass 

    def abs_summarize(self):
        pass 

    def clean_src(self, src_docs):
        pass 