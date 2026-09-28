from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

import torch
import soundfile as sf
import gc
import os

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from IndicTransToolkit.processor import IndicProcessor
from parler_tts import ParlerTTSForConditionalGeneration


# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(__name__)

CORS(app)


# ============================================================
# DEVICES
# ============================================================

# IndicTrans2
# Use GPU if CUDA is available

translation_device = (
    "cuda:0"
    if torch.cuda.is_available()
    else "cpu"
)


# Parler-TTS
# Your GTX 1650 has limited VRAM.
# Keep TTS on CPU.

tts_device = "cpu"


# ============================================================
# PROJECT INFORMATION
# ============================================================

print("====================================")
print(" Hindi → Santhali AI Translator")
print("====================================")

print(
    "\nTranslation device:",
    translation_device
)

print(
    "TTS device:",
    tts_device
)


# ============================================================
# INDIC TRANS2
# Hindi → Santhali
# ============================================================

TRANSLATION_MODEL = (
    "ai4bharat/indictrans2-indic-indic-dist-320M"
)


print("\nLoading IndicTrans2 tokenizer...")


translation_tokenizer = (
    AutoTokenizer.from_pretrained(
        TRANSLATION_MODEL,
        trust_remote_code=True
    )
)


print("Loading IndicTrans2 model...")


translation_model = (
    AutoModelForSeq2SeqLM.from_pretrained(
        TRANSLATION_MODEL,
        trust_remote_code=True
    )
    .to(translation_device)
)


translation_processor = IndicProcessor(
    inference=True
)


print("IndicTrans2 loaded successfully!")


# ============================================================
# INDIC PARLER-TTS
# Santhali Text → Santhali Speech
# ============================================================

TTS_MODEL = "ai4bharat/indic-parler-tts"


print("\nLoading Indic Parler-TTS...")

print(
    "TTS is running on CPU "
    "to avoid GTX 1650 VRAM overflow."
)


tts_model = (
    ParlerTTSForConditionalGeneration
    .from_pretrained(TTS_MODEL)
    .to(tts_device)
)


print("Indic Parler-TTS loaded successfully!")


# ============================================================
# TTS TOKENIZERS
# ============================================================

print("\nLoading TTS tokenizers...")


tts_tokenizer = AutoTokenizer.from_pretrained(
    TTS_MODEL
)


description_tokenizer = AutoTokenizer.from_pretrained(
    tts_model.config.text_encoder._name_or_path
)


print("TTS tokenizers loaded!")


# ============================================================
# TRANSLATION FUNCTION
# Hindi → Santhali
# ============================================================

def translate_to_santhali(sentence):

    print("\nTranslating Hindi → Santhali...")


    # --------------------------------------------------------
    # Language codes
    # --------------------------------------------------------

    src_lang = "hin_Deva"

    tgt_lang = "sat_Olck"


    # --------------------------------------------------------
    # Prepare input
    # --------------------------------------------------------

    batch = translation_processor.preprocess_batch(
        [sentence],
        src_lang=src_lang,
        tgt_lang=tgt_lang
    )


    # --------------------------------------------------------
    # Tokenization
    # --------------------------------------------------------

    inputs = translation_tokenizer(
        batch,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )


    # Move tensors to translation device

    inputs = inputs.to(
        translation_device
    )


    # --------------------------------------------------------
    # Translation
    # --------------------------------------------------------

    with torch.no_grad():

        generated_tokens = (
            translation_model.generate(
                **inputs,

                max_length=128,

                num_beams=1,

                num_return_sequences=1
            )
        )


    # --------------------------------------------------------
    # Decode
    # --------------------------------------------------------

    decoded = (
        translation_tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True
        )
    )


    # --------------------------------------------------------
    # Postprocess
    # --------------------------------------------------------

    output = (
        translation_processor.postprocess_batch(
            decoded,
            lang=tgt_lang
        )
    )


    santhali_text = output[0]


    print("Hindi    :", sentence)

    print(
        "Santhali :",
        santhali_text
    )


    return santhali_text


# ============================================================
# TTS FUNCTION
# Santhali Text → Santhali Speech
# ============================================================

def generate_santhali_speech(
    santhali_text
):

    print("\nGenerating Santhali speech...")

    print(
        "TTS input:",
        santhali_text
    )


    # --------------------------------------------------------
    # Voice description
    # --------------------------------------------------------

    description = (
        "A clear and natural speaker delivers "
        "the speech at a moderate speed. "
        "The voice is clear and easy to understand "
        "with good audio quality."
    )


    # --------------------------------------------------------
    # Description tokenizer
    # CPU
    # --------------------------------------------------------

    description_inputs = (
        description_tokenizer(
            description,
            return_tensors="pt"
        )
        .to(tts_device)
    )


    # --------------------------------------------------------
    # Santhali text tokenizer
    # CPU
    # --------------------------------------------------------

    prompt_inputs = (
        tts_tokenizer(
            santhali_text,
            return_tensors="pt"
        )
        .to(tts_device)
    )


    # --------------------------------------------------------
    # Generate speech
    # --------------------------------------------------------

    print("Running Parler-TTS...")


    with torch.no_grad():

        generation = (
            tts_model.generate(

                input_ids=(
                    description_inputs.input_ids
                ),

                attention_mask=(
                    description_inputs.attention_mask
                ),

                prompt_input_ids=(
                    prompt_inputs.input_ids
                ),

                prompt_attention_mask=(
                    prompt_inputs.attention_mask
                )
            )
        )


    # --------------------------------------------------------
    # Convert audio
    # --------------------------------------------------------

    audio = (
        generation
        .cpu()
        .numpy()
        .squeeze()
    )


    # --------------------------------------------------------
    # Save audio
    # --------------------------------------------------------

    output_file = os.path.abspath(
        "santhali_output.wav"
    )


    sf.write(
        output_file,
        audio,
        tts_model.config.sampling_rate
    )


    print(
        "Audio saved:",
        output_file
    )


    # --------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------

    del description_inputs

    del prompt_inputs

    del generation

    del audio


    gc.collect()


    if torch.cuda.is_available():

        torch.cuda.empty_cache()


    return output_file


# ============================================================
# TRANSLATE API
#
# Hindi
#   ↓
# IndicTrans2
#   ↓
# Santhali
#
# IMPORTANT:
# TTS is NOT called here.
#
# This makes translation faster and prevents the UI
# from waiting for Parler-TTS.
# ============================================================

@app.route(
    "/translate",
    methods=["POST"]
)
def translate():

    print("\n====================================")
    print("             TRANSLATE")
    print("====================================")


    # --------------------------------------------------------
    # Read JSON
    # --------------------------------------------------------

    data = request.get_json(
        silent=True
    )


    # --------------------------------------------------------
    # Validate JSON
    # --------------------------------------------------------

    if not data:

        return jsonify({
            "error": "Invalid or missing JSON"
        }), 400


    if "text" not in data:

        return jsonify({
            "error": "No Hindi text provided"
        }), 400


    # --------------------------------------------------------
    # Get Hindi text
    # --------------------------------------------------------

    hindi_text = str(
        data["text"]
    ).strip()


    if not hindi_text:

        return jsonify({
            "error": "Hindi text is empty"
        }), 400


    print(
        "Received Hindi:",
        hindi_text
    )


    try:

        # ----------------------------------------------------
        # ONLY TRANSLATION
        # ----------------------------------------------------

        santhali_text = (
            translate_to_santhali(
                hindi_text
            )
        )


        print(
            "Translation completed."
        )


        # ----------------------------------------------------
        # Return translation
        # ----------------------------------------------------

        return jsonify({

            "hindi": hindi_text,

            "santhali": santhali_text

        }), 200


    except Exception as e:

        print(
            "\n❌ Translation error:"
        )

        print(
            repr(e)
        )


        gc.collect()


        if torch.cuda.is_available():

            torch.cuda.empty_cache()


        return jsonify({

            "error": str(e)

        }), 500


# ============================================================
# AUDIO API
#
# Santhali text
#       ↓
# Parler-TTS
#       ↓
# WAV
#
# POST /audio
# ============================================================

@app.route(
    "/audio",
    methods=["POST"]
)
def audio():

    print("\n====================================")
    print("               AUDIO")
    print("====================================")


    # --------------------------------------------------------
    # Read JSON
    # --------------------------------------------------------

    data = request.get_json(
        silent=True
    )


    if not data:

        return jsonify({

            "error":
            "Invalid or missing JSON"

        }), 400


    # --------------------------------------------------------
    # Get Santhali text
    # --------------------------------------------------------

    if "text" not in data:

        return jsonify({

            "error":
            "No Santhali text provided"

        }), 400


    santhali_text = str(
        data["text"]
    ).strip()


    if not santhali_text:

        return jsonify({

            "error":
            "Santhali text is empty"

        }), 400


    print(
        "Received Santhali:",
        santhali_text
    )


    try:

        # ----------------------------------------------------
        # Generate speech
        # ----------------------------------------------------

        audio_file = (
            generate_santhali_speech(
                santhali_text
            )
        )


        # ----------------------------------------------------
        # Check file
        # ----------------------------------------------------

        if not os.path.exists(
            audio_file
        ):

            raise Exception(
                "Audio file was not created."
            )


        if os.path.getsize(
            audio_file
        ) == 0:

            raise Exception(
                "Generated audio file is empty."
            )


        print(
            "Sending audio to browser..."
        )


        # ----------------------------------------------------
        # Send WAV
        # ----------------------------------------------------

        return send_file(

            audio_file,

            mimetype="audio/wav",

            as_attachment=False,

            download_name=(
                "santhali_output.wav"
            )

        )


    except Exception as e:

        print(
            "\n❌ TTS error:"
        )

        print(
            repr(e)
        )


        gc.collect()


        if torch.cuda.is_available():

            torch.cuda.empty_cache()


        return jsonify({

            "error":
            str(e)

        }), 500


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route(
    "/",
    methods=["GET"]
)
def home():

    return jsonify({

        "status": "running",

        "project":
        "Hindi → Santhali AI Translator",

        "pipeline": [

            "Hindi Speech",

            "Vosk",

            "Hindi Text",

            "IndicTrans2",

            "Santhali Text",

            "Indic Parler-TTS",

            "Santhali Speech"

        ]

    }), 200


# ============================================================
# SERVER START
# ============================================================

if __name__ == "__main__":

    print("\n====================================")

    print(
        " Server starting..."
    )

    print(
        "===================================="
    )


    print("""
Pipeline:

Hindi Speech
     ↓
Vosk
     ↓
Hindi Text
     ↓
IndicTrans2
     ↓
Santhali Text
     ↓
Indic Parler-TTS
     ↓
Santhali Speech
""")


    print(
        "Translation device:",
        translation_device
    )

    print(
        "TTS device:",
        tts_device
    )


    print("\nServer URL:")

    print(
        "http://127.0.0.1:5000"
    )


    print(
        "http://localhost:5000"
    )


    app.run(

        host="0.0.0.0",

        port=5000,

        debug=False

    )
