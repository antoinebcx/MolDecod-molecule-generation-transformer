import sentencepiece as spm

def load_tokenizer(model_path):
    sp = spm.SentencePieceProcessor()
    sp.load(model_path)
    return sp
