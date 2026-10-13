import os
from huggingface_hub import InferenceClient

client = InferenceClient(
    provider="hf-inference",
    api_key=os.environ["HF_TOKEN"],
)

def translate(text: str) -> str:
    result = client.translation(
        text,
        model="facebook/nllb-200-distilled-600M",
        src_lang="eng_Latn",
        tgt_lang="tam_Taml",
    )
    return result.translation_text