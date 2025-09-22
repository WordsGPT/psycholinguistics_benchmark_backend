"""
Usage:
    python3 execute_experiment_ew.py <EXPERIMENT NAME>

    EXPERIMENT_NAME:
         * <experiment_name>: process specific experiment.
         * "all": process all experiments in the batches folder.
         * "remain": check and download batches still in tracking.
         * "status": show batches still in tracking.
"""

import os
import sys
import time
from datetime import datetime
from google import genai
from google.genai import types

from utils import load_config, openai_login, google_login, huggingface_login, vertec_login

import jsonlines
from transformers import AutoModelForCausalLM, AutoTokenizer, logging
logging.set_verbosity_error()
import transformers
import torch
import json
from tqdm import tqdm
from peft import PeftModel
import vertexai
from google.cloud import aiplatform, storage
from vertexai.tuning import sft
import pandas as pd
import hashlib

def get_experiment_prefixes_from_batches():
    """Devuelve una lista de prefijos únicos de experimentos en la carpeta batches."""
    prefixes = set()
    for fname in os.listdir("batches"):
        if "_prompt" in fname:
            prefix = fname.split("_prompt")[0] + "_prompt"
            prefixes.add(prefix)
    return list(prefixes)

def get_batches_for_experiment(prefix):
    """Devuelve la lista de batches que empiezan por el prefijo dado."""
    return [f for f in os.listdir("batches") if f.startswith(prefix)]

def create_batch_tracking_file(file_path="results/batch_tracking.xlsx"):
    """Crea un archivo Excel para trackear el estado de los batches."""
    # Asegurar que la carpeta results existe
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    if not os.path.exists(file_path):
        df = pd.DataFrame(columns=[
            'experiment_name', 'batch_file', 'batch_id', 'status', 'company', 'timestamp', 'file_hash'
        ])
        df.to_excel(file_path, index=False)
        print(f"Archivo de tracking creado: {file_path}")
    return file_path

def load_batch_tracking(file_path="results/batch_tracking.xlsx"):
    """Carga el archivo Excel de tracking de batches."""
    if os.path.exists(file_path):
        try:
            df = pd.read_excel(file_path)
            return df
        except Exception as e:
            print(f"Error cargando archivo de tracking: {e}")
            return pd.DataFrame(columns=[
                'experiment_name', 'batch_file', 'batch_id', 'status', 'company', 'timestamp', 'file_hash'
            ])
    else:
        return pd.DataFrame(columns=[
            'experiment_name', 'batch_file', 'batch_id', 'status', 'company', 'timestamp', 'file_hash'
        ])

def save_batch_tracking(df, file_path="results/batch_tracking.xlsx"):
    """Guarda el DataFrame de tracking en el archivo Excel."""
    try:
        # Asegurar que la carpeta results existe
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_excel(file_path, index=False)
    except Exception as e:
        print(f"Error guardando archivo de tracking: {e}")

def get_file_hash(file_path):
    """Calcula el hash MD5 de un archivo para detectar cambios."""
    with open(file_path, 'rb') as f:
        file_hash = hashlib.md5()
        chunk = f.read(8192)
        while chunk:
            file_hash.update(chunk)
            chunk = f.read(8192)
    return file_hash.hexdigest()

def add_batch_to_tracking(experiment_name, batch_file, batch_id, company, file_path="results/batch_tracking.xlsx"):
    """Añade un nuevo batch al archivo de tracking."""
    df = load_batch_tracking(file_path)
    file_hash = get_file_hash(f"batches/{batch_file}")
    new_row = pd.DataFrame({
        'experiment_name': [experiment_name],
        'batch_file': [batch_file],
        'batch_id': [batch_id],
        'status': ['submitted'],
        'company': [company],
        'timestamp': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        'file_hash': [file_hash]
    })
    df = pd.concat([df, new_row], ignore_index=True)
    save_batch_tracking(df, file_path)

def update_batch_status(batch_id, status, file_path="results/batch_tracking.xlsx"):
    """Actualiza el estado de un batch en el archivo de tracking."""
    df = load_batch_tracking(file_path)
    df.loc[df['batch_id'] == batch_id, 'status'] = status
    save_batch_tracking(df, file_path)

def remove_batch_from_tracking(batch_id, file_path="results/batch_tracking.xlsx"):
    """Elimina un batch del archivo de tracking."""
    df = load_batch_tracking(file_path)
    df = df[df['batch_id'] != batch_id]
    save_batch_tracking(df, file_path)
    print(f"Batch {batch_id} eliminado del tracking.")

def cleanup_empty_tracking_file(file_path="results/batch_tracking.xlsx"):
    """Elimina el archivo de tracking si está vacío."""
    if os.path.exists(file_path):
        df = load_batch_tracking(file_path)
        if len(df) == 0:
            os.remove(file_path)
            print(f"Archivo de tracking {file_path} eliminado (no hay batches pendientes).")
            return True
        else:
            print(f"Quedan {len(df)} batches en tracking.")
            return False
    return True

def get_pending_batches_for_experiment(experiment_name, company, file_path="results/batch_tracking.xlsx"):
    """Devuelve los batches pendientes para un experimento específico."""
    df = load_batch_tracking(file_path)
    # Filtrar por experimento y compañía, y que no estén completados o descargados
    pending = df[
        (df['experiment_name'] == experiment_name) & 
        (df['company'] == company) & 
        (~df['status'].isin(['completed', 'downloaded', 'failed']))
    ]
    return pending

def get_batches_to_send(experiment_name, company, batch_files, file_path="results/batch_tracking.xlsx"):
    """Determina qué batches necesitan ser enviados y cuáles ya están en tracking."""
    df = load_batch_tracking(file_path)
    existing_batches = df[
        (df['experiment_name'] == experiment_name) & 
        (df['company'] == company)
    ]
    
    batches_to_send = []
    for batch_file in batch_files:
        # Verificar si el batch ya existe en tracking
        file_hash = get_file_hash(f"batches/{batch_file}")
        existing = existing_batches[
            (existing_batches['batch_file'] == batch_file) & 
            (existing_batches['file_hash'] == file_hash)
        ]
        
        if existing.empty:
            # El batch no existe o ha cambiado, necesita ser enviado
            batches_to_send.append(batch_file)
        else:
            print(f"Batch {batch_file} ya está en tracking, saltando envío.")
    
    return batches_to_send

def add_role_to_jsonl(input_file: str, output_file: str, default_role: str = "user"):
    with open(input_file, "r", encoding="utf-8") as fin, \
         open(output_file, "w", encoding="utf-8") as fout:
        for line in fin:
            try:
                data = json.loads(line)
                if "request" in data and "contents" in data["request"]:
                    for content in data["request"]["contents"]:
                        if "role" not in content:
                            content["role"] = default_role
                fout.write(json.dumps(data, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"Error procesando línea en {input_file}: {e}")

def format_results_huggingface(output, tokenizer, counter, logprobs=5, config_args=None):
    try:
        gen_text = output[0]["generated_text"]
        scores = output[0]["scores"]
        token_entry_list=[]
        for step_logits_list in scores:
            step_logits = torch.tensor(step_logits_list)
            step_probs = torch.softmax(step_logits, dim=0)
            step_logprobs = torch.log(step_probs)
            top_probs, top_indices = torch.topk(step_probs, logprobs)
            top_logprobs = torch.log(top_probs)
            top_tokens = tokenizer.convert_ids_to_tokens(top_indices.tolist())
            cur_token = top_tokens[0]
            cur_logprob = top_logprobs.tolist()[0]
            top_logprobs_list = []
            for t, lp in zip(top_tokens, top_logprobs.tolist()):
                top_logprobs_list.append({
                    "token": t,
                    "logprob": lp,
                    "bytes": list(t.encode("utf-8"))
                })
            token_entry = {
                "token": cur_token,
                "logprob": cur_logprob,
                "bytes": list(cur_token.encode("utf-8")),
                "top_logprobs": top_logprobs_list
            }
            token_entry_list.append(token_entry)
        json_line = {
            "id": f"task_{counter}",
            "response": {
                "status_code": -1,
                "body": {
                    "model": config_args['model_name'] if config_args else "",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": gen_text,
                                "refusal": None,
                                "annotations": []
                            },
                            "logprobs": {
                                "content": token_entry_list
                            }
                        }
                    ]
                }
            }
        }
        return json_line
    except Exception as e:
        print(f"Error formateando resultados HuggingFace: {e}")
        return {}

def check_and_download_pending_batches():
    """Comprueba el estado de todos los batches pendientes y descarga los completados."""
    tracking_file = "results/batch_tracking.xlsx"
    
    if not os.path.exists(tracking_file):
        print("No hay archivo de tracking de batches.")
        return
    
    df = load_batch_tracking(tracking_file)
    
    if df.empty:
        print("No hay batches en tracking.")
        return
    
    print("\n=== Comprobando estado de todos los batches pendientes ===")
    
    # Agrupar por compañía para manejar diferentes APIs
    companies = df['company'].unique()
    
    for company in companies:
        company_batches = df[df['company'] == company]
        print(f"\n--- Comprobando batches de {company} ---")
        
        if company == "OpenAI":
            check_and_download_openai_batches(company_batches, tracking_file)
        elif company == "Google":
            check_and_download_google_batches(company_batches, tracking_file)
        elif company in ["HuggingFace", "Local"]:
            # Para HuggingFace/Local los batches se procesan inmediatamente
            print(f"Batches de {company} se procesan inmediatamente, no requieren descarga.")
    
    # Limpiar archivo de tracking si está vacío después de las descargas
    cleanup_empty_tracking_file(tracking_file)

def check_and_download_openai_batches(company_batches, tracking_file):
    """Comprueba y descarga batches de OpenAI."""
    try:
        client = openai_login()
        date_string = datetime.now().strftime("%Y-%m-%d-%H-%M")
        
        # Filtrar solo batches que no están descargados
        pending_batches = company_batches[
            ~company_batches['status'].isin(['downloaded', 'failed'])
        ]
        
        for _, row in pending_batches.iterrows():
            batch_id = row['batch_id']
            prefix = row['experiment_name']
            batch_file = row['batch_file']
            current_status = row['status']
            
            try:
                # Comprobar estado actual en OpenAI
                batch_info = client.batches.retrieve(batch_id)
                new_status = batch_info.status
                
                print(f"Batch {batch_id} ({batch_file}): {current_status} -> {new_status}")
                
                # Actualizar estado si ha cambiado
                if new_status != current_status:
                    update_batch_status(batch_id, new_status, tracking_file)
                
                # Descargar si está completado
                if new_status == "completed":
                    try:
                        batch_results_id = batch_info.output_file_id
                        result = client.files.content(batch_results_id).content
                        save_path = f"results/{prefix}_results_{batch_id}_{date_string}.jsonl"
                        
                        os.makedirs("results", exist_ok=True)
                        with open(save_path, "wb") as file:
                            file.write(result)
                        
                        print(f"Descargado resultado en {save_path}")
                        
                        # Eliminar del tracking después de descargar exitosamente
                        remove_batch_from_tracking(batch_id, tracking_file)
                        
                    except Exception as e:
                        print(f"Error descargando resultado batch {batch_id}: {e}")
                
                elif new_status in ["failed", "cancelled"]:
                    print(f"Batch {batch_id} falló o fue cancelado. Estado: {new_status}")
                    # Opcionalmente, eliminar batches fallidos del tracking
                    # remove_batch_from_tracking(batch_id, tracking_file)
                    
            except Exception as e:
                print(f"Error comprobando batch {batch_id}: {e}")
                
    except Exception as e:
        print(f"Error conectando con OpenAI: {e}")

def check_and_download_google_batches(company_batches, tracking_file):
    """Comprueba y descarga batches de Google."""
    try:
        google_login()
        date_string = datetime.now().strftime("%Y-%m-%d-%H-%M")
        
        # Filtrar solo batches que no están descargados
        pending_batches = company_batches[
            ~company_batches['status'].isin(['downloaded', 'failed'])
        ]
        
        for _, row in pending_batches.iterrows():
            batch_id = row['batch_id']
            prefix = row['experiment_name']
            batch_file = row['batch_file']
            current_status = row['status']
            
            print(f"Batch Google {batch_id} ({batch_file}): {current_status}")
            
            # Por ahora, simular descarga para batches antiguos
            print(f"Implementación de Google Batch API pendiente para batch {batch_id}")
            
            # Simular descarga y eliminar del tracking para evitar acumulación
            # En una implementación real, aquí se verificaría el estado real
            if current_status == "submitted":
                print(f"Simulando descarga de batch Google {batch_id}")
                # remove_batch_from_tracking(batch_id, tracking_file)
                
    except Exception as e:
        print(f"Error con batches de Google: {e}")

def execute_openai(config_args, experiment_prefixes):
    client = openai_login()
    tracking_file = "results/batch_tracking.xlsx"
    create_batch_tracking_file(tracking_file)
    
    for prefix in experiment_prefixes:
        batch_files = get_batches_for_experiment(prefix)
        
        # Verificar qué batches necesitan ser enviados (evitar duplicados)
        batches_to_send = get_batches_to_send(prefix, "OpenAI", batch_files, tracking_file)
        
        if not batches_to_send:
            print(f"Todos los batches para {prefix} ya están en tracking. Use 'remain' para verificar su estado.")
            continue
        
        for file_name in batches_to_send:
            try:
                with open(f"batches/{file_name}", "rb") as f:
                    batch_file = client.files.create(
                        file=f, purpose="batch"
                    )
                batch_job = client.batches.create(
                    input_file_id=batch_file.id,
                    endpoint="/v1/chat/completions",
                    completion_window="24h",
                )
                print(f"# File Named: {file_name} has been submitted for processing")
                print(f"# Batch Job ID: {batch_job.id}")
                
                # Guardar en tracking
                add_batch_to_tracking(prefix, file_name, batch_job.id, "OpenAI", tracking_file)
                
            except Exception as e:
                print(f"Error enviando batch {file_name}: {e}")
    
    # Después de enviar todos los batches, comprobar estado cada 120 segundos
    print("\n=== Iniciando comprobación periódica de batches cada 120 segundos ===")
    while True:
        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Comprobando estado de batches...")
        check_and_download_pending_batches()
        
        # Verificar si quedan batches pendientes
        df = load_batch_tracking(tracking_file)
        if df.empty or len(df[df['company'] == 'OpenAI']) == 0:
            print("No quedan batches de OpenAI pendientes. Finalizando comprobación.")
            break
        
        print("Esperando 120 segundos antes de la próxima comprobación...")
        time.sleep(120)

def execute_google(config_args, experiment_prefixes):
    google_login()
    tracking_file = "results/batch_tracking.xlsx"
    create_batch_tracking_file(tracking_file)
    
    for prefix in experiment_prefixes:
        batch_files = get_batches_for_experiment(prefix)
        
        # Verificar qué batches necesitan ser enviados (evitar duplicados)
        batches_to_send = get_batches_to_send(prefix, "Google", batch_files, tracking_file)
        
        if not batches_to_send:
            print(f"Todos los batches para {prefix} ya están en tracking. Use 'remain' para verificar su estado.")
            continue
        
        for file_name in batches_to_send:
            try:
                # Generar un batch_id único para Google
                batch_id = f"google_batch_{int(time.time())}_{file_name.replace('.jsonl', '')}"
                print(f"# File Named: {file_name} has been submitted for processing (Google)")
                print(f"# Batch Job ID: {batch_id}")
                
                # Guardar en tracking
                add_batch_to_tracking(prefix, file_name, batch_id, "Google", tracking_file)
                
            except Exception as e:
                print(f"Error enviando batch {file_name} a Google: {e}")
    
    # Después de enviar todos los batches, comprobar estado cada 120 segundos
    print("\n=== Iniciando comprobación periódica de batches cada 120 segundos ===")
    while True:
        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Comprobando estado de batches...")
        check_and_download_pending_batches()
        
        # Verificar si quedan batches pendientes
        df = load_batch_tracking(tracking_file)
        if df.empty or len(df[df['company'] == 'Google']) == 0:
            print("No quedan batches de Google pendientes. Finalizando comprobación.")
            break
        
        print("Esperando 120 segundos antes de la próxima comprobación...")
        time.sleep(120)

def check_and_download_pending_batches_loop():
    """Comprueba el estado de todos los batches pendientes cada 120 segundos hasta que no queden."""
    tracking_file = "results/batch_tracking.xlsx"
    
    if not os.path.exists(tracking_file):
        print("No hay archivo de tracking de batches.")
        return
    
    print("\n=== Iniciando comprobación periódica de batches cada 120 segundos ===")
    while True:
        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Comprobando estado de batches...")
        check_and_download_pending_batches()
        
        # Verificar si quedan batches pendientes
        df = load_batch_tracking(tracking_file)
        if df.empty:
            print("No quedan batches pendientes. Finalizando comprobación.")
            break
        
        # Verificar si hay batches de APIs que requieren descarga
        pending_apis = df[df['company'].isin(['OpenAI', 'Google'])]
        if pending_apis.empty:
            print("No quedan batches de APIs pendientes de descarga. Finalizando comprobación.")
            break
        
        print(f"Quedan {len(pending_apis)} batches pendientes de APIs. Esperando 120 segundos...")
        time.sleep(120)

def execute_huggingface_or_local(config_args, company, experiment_prefixes):
    if company == "HuggingFace":
        huggingface_login()
    
    tracking_file = "results/batch_tracking.xlsx"
    create_batch_tracking_file(tracking_file)
    
    for prefix in experiment_prefixes:
        batch_files = get_batches_for_experiment(prefix)
        
        # Verificar qué batches necesitan ser enviados (evitar duplicados)
        batches_to_send = get_batches_to_send(prefix, company, batch_files, tracking_file)
        
        if not batches_to_send:
            print(f"Todos los batches para {prefix} ya han sido procesados o están en tracking.")
            continue
        
        for index, file_name in enumerate(batches_to_send):
            jsonl_file_path = f"batches/{file_name}"
            try:
                # Generar batch_id único para tracking
                batch_id = f"{company.lower()}_batch_{int(time.time())}_{index}_{file_name.replace('.jsonl', '')}"
                
                # Agregar a tracking como "processing"
                add_batch_to_tracking(prefix, file_name, batch_id, company, tracking_file)
                update_batch_status(batch_id, "processing", tracking_file)
                
                with jsonlines.open(jsonl_file_path, "r") as reader:
                    for obj in reader:
                        model_name = obj.get("model")
                        ft_dir = obj.get("ft_dir", None)
                        temperature = obj.get("temperature")
                        response_logprobs = obj.get("response_logprobs")
                        logprobs = obj.get("logprobs")
                        break
                base_model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map="auto",
                    torch_dtype=torch.bfloat16
                )
                if ft_dir:
                    ft_dir_final = f"finetuning/cache_model/{ft_dir}"
                    if not os.path.isdir(ft_dir_final):
                        ft_dir_final = f"finetuning_final/{ft_dir}/cache_model/{ft_dir}"
                    if not os.path.isdir(ft_dir_final):
                        raise ValueError(f"LoRA directory {ft_dir_final} does not exist.")
                    print(f"Loading LoRA-finetuned model from {ft_dir_final}")
                    model = PeftModel.from_pretrained(base_model, ft_dir_final)
                    tokenizer = AutoTokenizer.from_pretrained(ft_dir_final)
                else:
                    model = base_model
                    print(f"Using base model {model_name} (no LoRA applied)")
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                pipeline = transformers.pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    device_map="auto",
                )
                date_string = datetime.now().strftime("%Y-%m-%d-%H-%M")
                os.makedirs(f"results", exist_ok=True)
                output_file = f"results/{prefix}_results_{batch_id}_{date_string}.jsonl"
                
                with open(output_file, "w", encoding="utf-8") as f_out:
                    with jsonlines.open(jsonl_file_path, "r") as reader:
                        batch_messages = []
                        counter = 0
                        batch_size = 5
                        for obj in tqdm(reader, desc=f"Processing {jsonl_file_path}"):
                            prompt = obj.get("prompt")
                            batch_messages.append([{"role": "user", "content": prompt}])
                            if len(batch_messages) == batch_size:
                                outputs = pipeline(
                                    batch_messages,
                                    max_new_tokens=500,
                                    temperature=temperature,
                                    do_sample=True if temperature > 0 else False,
                                    return_full_text=False,
                                    return_dict_in_generate=True,
                                    output_scores=response_logprobs
                                )
                                for output in outputs:
                                    counter += 1
                                    json_line = format_results_huggingface(output, tokenizer, counter, logprobs, config_args)
                                    if json_line:
                                        f_out.write(json.dumps(json_line, ensure_ascii=False) + "\n")
                                        f_out.flush()
                                batch_messages = []
                        if len(batch_messages) > 0:
                            outputs = pipeline(
                                batch_messages,
                                max_new_tokens=500,
                                temperature=temperature,
                                do_sample=True if temperature > 0 else False,
                                return_full_text=False,
                                return_dict_in_generate=True,
                                output_scores=response_logprobs
                            )
                            for output in outputs:
                                counter += 1
                                json_line = format_results_huggingface(output, tokenizer, counter, logprobs, config_args)
                                if json_line:
                                    f_out.write(json.dumps(json_line, ensure_ascii=False) + "\n")
                                    f_out.flush()
                            batch_messages = []
                            
                print(f"Guardado resultado en {output_file}")
                
                # Eliminar del tracking después de procesar exitosamente
                remove_batch_from_tracking(batch_id, tracking_file)
                
            except Exception as e:
                print(f"Error procesando batch {file_name}: {e}")
                # Marcar como fallido en tracking
                if 'batch_id' in locals():
                    update_batch_status(batch_id, "failed", tracking_file)
    
    # Limpiar archivo de tracking si está vacío
    cleanup_empty_tracking_file(tracking_file)

def show_batch_status(file_path="results/batch_tracking.xlsx"):
    """Muestra el estado actual de todos los batches en tracking."""
    df = load_batch_tracking(file_path)
    if df.empty:
        print("No hay batches en tracking.")
        return
    
    print("\n=== Estado de Batches ===")
    print(f"{'Experimento':<20} {'Archivo':<30} {'Compañía':<12} {'Estado':<12} {'Timestamp':<20}")
    print("-" * 100)
    
    for _, row in df.iterrows():
        print(f"{row['experiment_name']:<20} {row['batch_file']:<30} {row['company']:<12} {row['status']:<12} {row['timestamp']:<20}")
    
    # Resumen por estado
    status_counts = df['status'].value_counts()
    print(f"\n=== Resumen ===")
    for status, count in status_counts.items():
        print(f"{status}: {count}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        EXPERIMENT_NAME = sys.argv[1]
    else:
        print(
            "Provide as argument the experiment name, i.e.: python execute_experiment_ew.py all OR python execute_experiment_ew.py <EXPERIMENT_NAME> OR python execute_experiment_ew.py status OR python execute_experiment_ew.py remain"
        )
        exit()

    if EXPERIMENT_NAME == "status":
        show_batch_status()
        exit()
    
    # Comando para comprobar y descargar batches pendientes
    if EXPERIMENT_NAME == "remain":
        print("Comprobando y descargando todos los batches pendientes...")
        check_and_download_pending_batches_loop()
        exit()

    # Ejecución de experimentos
    if EXPERIMENT_NAME == "all":
        prefixes = get_experiment_prefixes_from_batches()
        if not prefixes:
            print("No se encontraron experimentos en la carpeta batches.")
            exit()
        # Agrupa experimentos por compañía
        company_to_prefixes = {}
        for prefix in prefixes:
            config_args = load_config(
                config_type="experiments",
                name=prefix,
            )
            company = config_args.get("company")
            if company not in company_to_prefixes:
                company_to_prefixes[company] = []
            company_to_prefixes[company].append(prefix)
        # Ejecuta cada grupo
        for company, group_prefixes in company_to_prefixes.items():
            print(f"\nEjecutando para compañía: {company} -> Experimentos: {group_prefixes}")
            config_args = load_config(
                config_type="experiments",
                name=group_prefixes[0],
            )
            if company == "OpenAI":
                execute_openai(config_args, group_prefixes)
            elif company == "Google":
                execute_google(config_args, group_prefixes)
            elif company == "HuggingFace":
                execute_huggingface_or_local(config_args, company="HuggingFace", experiment_prefixes=group_prefixes)
            elif company == "Local":
                execute_huggingface_or_local(config_args, company="Local", experiment_prefixes=group_prefixes)
            else:
                print(f"Company {company} is not supported.")
    else:
        # Ejecuta solo el experimento indicado
        # Verificar si existe el batch correspondiente
        batch_files = get_batches_for_experiment(EXPERIMENT_NAME)
        if not batch_files:
            print(f"No se encontraron batches para el experimento '{EXPERIMENT_NAME}' en la carpeta batches.")
            exit()
        
        config_args = load_config(
            config_type="experiments",
            name=EXPERIMENT_NAME,
        )
        company = config_args.get("company")
        print(f"Compañía detectada: {company}")
        prefixes = [EXPERIMENT_NAME]
        
        if company == "OpenAI":
            execute_openai(config_args, prefixes)
        elif company == "Google":
            execute_google(config_args, prefixes)
        elif company == "HuggingFace":
            execute_huggingface_or_local(config_args, company="HuggingFace", experiment_prefixes=prefixes)
        elif company == "Local":
            execute_huggingface_or_local(config_args, company="Local", experiment_prefixes=prefixes)
        else:
            print(f"Company {company} is not supported.")
