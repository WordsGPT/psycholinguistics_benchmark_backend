import glob
import sys
import os
import subprocess
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import read_yaml

# Obtener el comando a ejecutar
if __name__ == "__main__":
    if len(sys.argv) > 1:
        SCRIPT_NAME = sys.argv[1]
    else:
        print(
            "Provide as argument the script name wanted to be executed, i.e.: python3 run_scripts_ew.py <SCRIPT_NAME>."
        )
        exit()

# Cargar el archivo config.yaml
config = read_yaml("config.yaml")

# Obtener la lista de nombres de experimentos
experiments = list(config.get("experiments", {}).keys())

FAILED_FILE = os.path.join("failed_exp", f"failed_{SCRIPT_NAME}_experiments.txt")

# Si existe el archivo de fallidos, solo ejecuta esos
if os.path.exists(FAILED_FILE):
    with open(FAILED_FILE, "r", encoding="utf-8") as f:
        failed_from_last = [line.strip() for line in f if line.strip()]
    if failed_from_last:
        print(f"Reintentando solo los experimentos fallidos: {', '.join(failed_from_last)}")
        experiments = failed_from_last

successful_experiments = 0
failed_experiments = []

for name in experiments:
    print(f"\nEjecutando: py {SCRIPT_NAME} {name}")

    matches_batches = glob.glob(f"batches/*{name}*.jsonl")
    matches_results = glob.glob(f"results/*{name}*.jsonl")

    if SCRIPT_NAME == "execute_experiment_ew.py":
        if not matches_batches:
            print(f"No batches file found for experiment name '{name}' in 'batches' folder.")
            failed_experiments.append(name)
            continue
    
    if SCRIPT_NAME == "generateResults_ew.py":
        if not matches_results and not matches_batches:
            print(f"No results file found for experiment name '{name}' in 'results' folder.")
            failed_experiments.append(name)
            continue

    result = subprocess.run(
        [sys.executable, SCRIPT_NAME, name],
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )
    if result.returncode != 0:
        print(f"Error ejecutando el experimento: {name}")
        failed_experiments.append(name)
    else:
        successful_experiments += 1


print(f"\n\nNúmero de experimentos ejecutados correctamente: {successful_experiments}\n\n")

if failed_experiments:
    print("Experimentos que no se han podido ejecutar: " + ", ".join(failed_experiments) + "\n\n")
    # Crear la carpeta si no existe
    os.makedirs("failed_exp", exist_ok=True)
    # Guardar los fallidos para la próxima ejecución
    with open(FAILED_FILE, "w", encoding="utf-8") as f:
        for name in failed_experiments:
            f.write(name + "\n")
else:
    # Si no hay fallidos, elimina el archivo si existe
    if os.path.exists(FAILED_FILE):
        os.remove(FAILED_FILE)