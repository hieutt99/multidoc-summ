import collections
import unicodedata
from io import open
from others.logging import logger
from tokenizers import BertWordPieceTokenizer
from transformers import AutoTokenizer



# logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

TOKENIZER_MAP = {
    "basic_ext": "bert-base-cased", 
    "basic_abs": "bert-base-cased", 
    "vi_basic_ext": "vinai/phobert-base",
    "vi_basic_abs": "vinai/phobert-base"
}

def build_tokenizer(args):
    return AutoTokenizer.from_pretrained(TOKENIZER_MAP[args.model_config.model_name], 
        cache_dir=args.temp_dir,
        use_fast=False
        )




def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    index = 0
    with open(vocab_file, "r", encoding="utf-8") as reader:
        while True:
            token = reader.readline()
            if not token:
                break
            token = token.strip()
            vocab[token] = index
            index += 1
    return vocab

def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a peice of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens

def _is_whitespace(char):
    """Checks whether `chars` is a whitespace character."""
    # \t, \n, and \r are technically contorl characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def _is_control(char):
    """Checks whether `chars` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False


def _is_punctuation(char):
    """Checks whether `chars` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
            (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False

def load_tokenizer(path=None):
    if path:
        try:
            tokenizer = BertWordPieceTokenizer.from_pretrained(path)
            return tokenizer 
        except:
            raise Exception("fail loading tokenizer")
    else:
        raise Exception("fail loading tokenizer as no path ")