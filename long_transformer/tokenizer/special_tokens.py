# class SpecialTokens():
#     sep_token = '[SEP]'
#     cls_token = '[CLS]'
#     pad_token = '[PAD]'
#     # bos_token = '[unused1]' # 0
#     # eos_token = '[unused2]' # 1
#     bos_token = '[SEP]' # 0
#     eos_token = '[CLS]' # 1
#     tgt_sent_split = '[unused1]' # 2
    
#     # temp
#     sep_vid = 102
#     cls_vid = 101
#     pad_vid = 0
#     tgt_sent_split_vid = 1

#     # tgt_sent_split = '[unused3]' # 2
#     # src_story_split = '[unused4]' # 3
#     # vocab = tokenizer.get_vocab()
#     # sep_vid = vocab[sep_token]
#     # cls_vid = vocab[cls_token]
#     # pad_vid = vocab[pad_token]


class SpecialTokens():
    sep_token = '</s>'
    cls_token = '<s>'
    pad_token = '<pad>'
    bos_token = '<s>' # 0
    eos_token = '</s>' # 
    additional_special_tokens = ['<ss>', '<ds>']


# class SpecialTokens():
#     sep_token = '[SEP]'
#     cls_token = '[CLS]'
#     pad_token = '[PAD]'
#     bos_token = '[CLS]' # 0
#     eos_token = '[SEP]' # 1
#     additional_special_tokens = ['[unused1]', '[unused2]']