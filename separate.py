# separate.py
import sys
from pathlib import Path
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from audio_separator.separator import Separator

if len(sys.argv) < 3:
    print("Uso: python separate.py <input> <modelo>")
    sys.exit(1)

INPUT = sys.argv[1]
filename = sys.argv[2]

MODELS_DIR = Path("./models_cache").resolve()
OUTPUT_DIR = Path("./stems").resolve()

sep = Separator(
    model_file_dir=str(MODELS_DIR),
    output_dir=str(OUTPUT_DIR),
    output_format="WAV",
    sample_rate=44100,
    use_autocast=False,
)

sep.download_model_files(filename)
sep.load_model(filename)
sep.separate(INPUT)

print(f"✅ Separation completed. See: {OUTPUT_DIR}")
