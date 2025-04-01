import sounddevice as sd
import numpy as np
import json
from vosk import Model, KaldiRecognizer

MODEL_PATH = "/Users/devayushrout/Desktop/MedWaste Guardian/stt/vosk-model-en-in-0.5"

def recognize_speech():
    """Captures audio from the microphone and performs speech-to-text."""

    model = Model(MODEL_PATH)
    recognizer = KaldiRecognizer(model, 16000)

    def callback(indata, frames, time, status):
        """Processes audio stream and sends it to Vosk for recognition."""
        if status:
            print(status)  

        if recognizer.AcceptWaveform(indata.tobytes()):
            result = json.loads(recognizer.Result())
            if "text" in result and result["text"].strip() != "":
                print("\nRecognized Text:", result["text"])

    print("\n🎤 Listening... Speak now. Press Ctrl+C to stop.")

    try:
        with sd.InputStream(samplerate=16000, channels=1, dtype="int16", callback=callback):
            while True:
                pass 

    except KeyboardInterrupt:
        print("\n Speech recognition stopped.")

if __name__ == "__main__":
    recognize_speech()
