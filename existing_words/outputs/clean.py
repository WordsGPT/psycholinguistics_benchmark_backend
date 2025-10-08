import os
import pandas as pd

# Carpeta base = donde estás
base_dir = os.getcwd()

# Recorremos las carpetas que terminan en '_prompt'
for root, dirs, files in os.walk(base_dir):
    if root.endswith("_prompt"):
        for file in files:
            if file.endswith(".xlsx"):
                file_path = os.path.join(root, file)
                try:
                    # Leer excel
                    df = pd.read_excel(file_path)

                    # Renombrar columna exist si existe
                    if "exist" in df.columns:
                        df = df.rename(columns={"exist": "exist_gpt-4o-mini-2024-07-18"})

                    # Borrar columnas si existen
                    for col in ["logprob", "weighted_sum"]:
                        if col in df.columns:
                            df = df.drop(columns=[col])

                    # Guardar sobre el mismo archivo
                    df.to_excel(file_path, index=False)
                    print(f"Procesado: {file_path}")

                except Exception as e:
                    print(f"Error con {file_path}: {e}")
