import threading
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "facebook/nllb-200-distilled-600M"
model_folder = "model/" + model_name.replace("/","_")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model.eval()
_lock = threading.Lock()

def translate(text: str) -> str:
    src_lang = "eng_Latn"
    targ_lang = "tam_Taml"
    tgt_id = tokenizer.convert_tokens_to_ids(targ_lang)
    if tgt_id == tokenizer.unk_token_id:
        raise ValueError(f"Unknown tgt_lang: {targ_lang}")
    if tokenizer.convert_tokens_to_ids(src_lang) == tokenizer.unk_token_id:
        raise ValueError(f"Unknown src_lang: {src_lang}")

    with _lock:
        tokenizer.src_lang = src_lang
        inputs = tokenizer(text, return_tensors = "pt")
        with torch.no_grad():
            output = model.generate(**inputs, forced_bos_token_id = tgt_id, max_new_tokens = 200)
        return  tokenizer.batch_decode(output, skip_special_tokens = True)[0]

