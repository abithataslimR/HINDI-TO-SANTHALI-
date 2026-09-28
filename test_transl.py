import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from IndicTransToolkit.processor import IndicProcessor

MODEL_NAME = "ai4bharat/indictrans2-indic-indic-dist-320M"

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

print("Loading model...")
model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
).to(device)

processor = IndicProcessor(inference=True)

# Get Hindi input from user
sentence = input("\nEnter Hindi sentence: ")

src_lang = "hin_Deva"
tgt_lang = "sat_Olck"

# Prepare input
batch = processor.preprocess_batch(
    [sentence],
    src_lang=src_lang,
    tgt_lang=tgt_lang
)

inputs = tokenizer(
    batch,
    padding=True,
    truncation=True,
    return_tensors="pt"
).to(device)

# Translate
with torch.no_grad():
    generated_tokens = model.generate(
        **inputs,
        max_length=256,
        num_beams=5,
        num_return_sequences=1
    )

# Decode
decoded = tokenizer.batch_decode(
    generated_tokens,
    skip_special_tokens=True
)

output = processor.postprocess_batch(
    decoded,
    lang=tgt_lang
)

print("\nHindi   :", sentence)
print("Santhali:", output[0])
