import whisper
import torch


# ------------------------------
# Load Whisper model ONCE
# ------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

whisper_model = whisper.load_model("base").to(DEVICE)


# ------------------------------
# ASR function
# ------------------------------
def run_whisper_asr(audio_path: str) -> str:
    """
    Run Whisper ASR on enhanced speech audio.

    Args:
        audio_path (str): Path to enhanced wav file

    Returns:
        str: Transcribed text
    """

    result = whisper_model.transcribe(audio_path)
    text = result["text"]

    return text


# ------------------------------
# Optional CLI usage (testing)
# ------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to enhanced wav file"
    )

    args = parser.parse_args()

    transcription = run_whisper_asr(args.input)
    print("\n[Whisper Transcription]")
    print(transcription)
