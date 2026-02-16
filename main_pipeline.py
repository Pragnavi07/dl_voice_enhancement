import argparse
from infer_enhance import enhance_audio

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True, help="Path to noisy audio file")
args = parser.parse_args()

enhance_audio(args.input)
