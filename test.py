import sounddevice as sd
import json
from vosk import Model, KaldiRecognizer

# Windows Vosk model accessed from WSL
MODEL_PATH = "/mnt/c/Users/Abitha Taslim/vosk-test/vosk-model-small-hi-0.22"

print("Loading Hindi Vosk model...")

model = Model(MODEL_PATH)


def recognize_speech():

    recognizer = KaldiRecognizer(model, 16000)

    print("\n🎤 Speak in Hindi...")
    print("Press Ctrl+C after speaking.\n")

    recognized_text = ""

    def callback(indata, frames, time, status):

        nonlocal recognized_text

        if status:
            print(status)

        audio_data = bytes(indata)

        if recognizer.AcceptWaveform(audio_data):

            result = json.loads(recognizer.Result())
            text = result.get("text", "")

            if text:
                recognized_text = text
                print("Hindi:", text)

    try:

        with sd.RawInputStream(
            samplerate=16000,
            blocksize=8000,
            dtype="int16",
            channels=1,
            callback=callback
        ):

            while True:
                sd.sleep(100)

    except KeyboardInterrupt:
        print("\nStopped listening.")

    return recognized_text


def translate_to_santhali(hindi_text):

    print("\n🧠 Sending Hindi text to IndicTrans2...")
    print("Input:", hindi_text)

    # IndicTrans2 will be connected here
    santhali_text = "TEMPORARY"

    return santhali_text


def voice_translate():

    hindi_text = recognize_speech()

    if not hindi_text:
        print("No Hindi speech detected.")
        return

    print("\nHindi Text:")
    print(hindi_text)

    santhali_text = translate_to_santhali(hindi_text)

    print("\nSanthali Translation:")
    print(santhali_text)


if __name__ == "__main__":
    voice_translate()
