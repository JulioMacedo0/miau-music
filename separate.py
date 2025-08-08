from pathlib import Path
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from audio_separator.separator import Separator

INPUT = "musica.mp3"
MODELS_DIR = Path("./models_cache").resolve()
OUTPUT_DIR = Path("./stems").resolve()

sep = Separator(
    model_file_dir=str(MODELS_DIR),
    output_dir=str(OUTPUT_DIR),
    output_format="WAV",
    sample_rate=44100,
    use_autocast=False,
)

# List available models
print("Available models:\n")
for model in sorted(sep.get_simplified_model_list().keys()):
    print(model)
print()

# Choose model manually

filename = "3_HP-Vocal-UVR.pth"
sep.download_model_files(filename)
sep.load_model(filename)
sep.separate(INPUT)

print(f"✅ Separation completed. See: {OUTPUT_DIR}")
