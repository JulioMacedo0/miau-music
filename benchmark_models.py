# benchmark_models.py
import csv
import time
from datetime import datetime
from pathlib import Path
import os
import subprocess

# === CONFIGURAÇÕES ===
INPUT = "musica.mp3"
OUTPUT_DIR = Path("./stems").resolve()
RUNS_CSV = Path("runs.csv").resolve()
FMT = "WAV"
SR = 44100

# Lista de modelos para teste
MODELS = [
    "10_SP-UVR-2B-32000-1.pth",
    "3_HP-Vocal-UVR.pth",
    "UVR-MDX-NET-Inst_HQ_3.onnx",
    "UVR-MDX-NET-Inst_Main.onnx",
]

# === PREPARA CSV ===
if not RUNS_CSV.exists():
    with open(RUNS_CSV, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp_utc", "input_file", "model_filename",
                "output_path", "output_name", "output_ext",
                "output_bytes", "output_dir", "output_format",
                "sample_rate", "proc_seconds", "status", "error"
            ]
        )
        w.writeheader()

# === FUNÇÃO PARA RODAR MODELO E LOGAR ===
def run_and_log(model: str):
    start = time.time()
    error = ""
    status = "ok"

    try:
        # Rodando o separate.py com o modelo escolhido
        subprocess.run(
            ["uv", "run", "python", "separate.py", INPUT, model],
            check=True
        )
    except subprocess.CalledProcessError as e:
        error = str(e)
        status = "fail"

    proc_seconds = time.time() - start
    timestamp = datetime.utcnow().isoformat(timespec="seconds") + "Z"

    # Lista todos os arquivos que contenham o nome do modelo no nome
    matching_files = [p for p in OUTPUT_DIR.glob("*") if model.replace(".pth", "").replace(".onnx", "") in p.name]

    # Se não achou, registra linha vazia
    if not matching_files:
        matching_files = []

    with open(RUNS_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "timestamp_utc", "input_file", "model_filename",
            "output_path", "output_name", "output_ext",
            "output_bytes", "output_dir", "output_format",
            "sample_rate", "proc_seconds", "status", "error"
        ])

        if matching_files:
            for out in matching_files:
                w.writerow({
                    "timestamp_utc": timestamp,
                    "input_file": str(Path(INPUT).resolve()),
                    "model_filename": model,
                    "output_path": str(out.resolve()),
                    "output_name": out.name,
                    "output_ext": out.suffix.lower(),
                    "output_bytes": out.stat().st_size,
                    "output_dir": str(OUTPUT_DIR),
                    "output_format": FMT,
                    "sample_rate": SR,
                    "proc_seconds": f"{proc_seconds:.3f}",
                    "status": status,
                    "error": error,
                })
        else:
            # Linha sem arquivo
            w.writerow({
                "timestamp_utc": timestamp,
                "input_file": str(Path(INPUT).resolve()),
                "model_filename": model,
                "output_path": "",
                "output_name": "",
                "output_ext": "",
                "output_bytes": 0,
                "output_dir": str(OUTPUT_DIR),
                "output_format": FMT,
                "sample_rate": SR,
                "proc_seconds": f"{proc_seconds:.3f}",
                "status": status,
                "error": error,
            })

# === EXECUTA TODOS OS MODELOS ===
for model in MODELS:
    print(f"\n🚀 Rodando modelo: {model}")
    run_and_log(model)

print("\n✅ Benchmark finalizado! Veja o arquivo:", RUNS_CSV)
