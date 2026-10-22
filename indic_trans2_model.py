import torch
import threading
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit.processor import IndicProcessor


DEVICE = "cuda" if torch.cuda.is_available else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

SRC_LANG  = "eng_Latn"
TGT_LANG  = "tam_Taml"
MODEL_NAME = "ai4bharat/indictrans2-en-indic-1B"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code = True)

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code = True,
    dtype = DTYPE
).to(DEVICE)
model.eval()

ip = IndicProcessor(inference = True)
_lock = threading.Lock()

def translate(text : str) -> str:
    return translate_batch([text])[0]

def translate_batch(texts: list[str]) -> list[str]:
    print(translate_batch)
    batch = ip.preprocess_batch(texts, src_lang = SRC_LANG,tgt_lang = TGT_LANG)

    with _lock:
        input = tokenizer(
            batch,
            truncation= True,
            padding="longest",
            return_tensors = "pt",
            return_attention_mask = True,
        ).to(DEVICE)

    with torch.no_grad():
        generate = model.generate(
            **input,
            use_cache = False,
            min_length = 0,
            max_length = 250,
            num_beams = 3,
            num_return_sequences = 1,
        )

    decode = tokenizer.batch_decode(
        generate,
        skip_special_tokens = True,
        clean_tokenization_cap = True,
    )

    return ip.postprocess_batch(decode, lang = TGT_LANG)