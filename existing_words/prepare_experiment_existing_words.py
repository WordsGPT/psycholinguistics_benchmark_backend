"""
This script is designed to prepare tasks for psycholinguistic experiments using the OpenAI API. It reads a configuration file and a list of words from an Excel file, then generates tasks formatted for the OpenAI API. These tasks are batched and saved as JSONL files for later execution.

Key Components:
- `load_word_list_from_excel`: Loads a list of words from an Excel file.
- `get_tasks`: Generates tasks for each word using a specified prompt and model configuration.
- `create_batches`: Splits tasks into batches and saves them as JSONL files.

Important Considerations:
1. Configuration File: Ensure that the `config_experiment.yaml` file is correctly set up with the necessary parameters `dataset_name`, `prompt_to_use`, and `model_name`.
2. Environment Variables: The `apis.env` file must contain valid OpenAI API credentials.
3. Excel File: The Excel file should be located in the specified experiment directory and contain a column with the words to be used in the experiment.

Usage:
Run the script from the command line with the experiment name as an argument:
    python prepare_experiment_existing_words.py 

Example:
    python prepare_experiment_existing_words.py my_experiment

This will create a directory named `my_experiment` with subdirectories for batches, containing the prepared task files.
"""

import json
import os
import sys
import jsonlines
from datetime import datetime

import pandas as pd

from utils import load_config, openai_login, read_txt


def load_prompt_list(file_path: str, column_name: str) -> list:
    print(f"Loading prompt list from {file_path} column {column_name}...")
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path, encoding='iso-8859-1')
    elif file_path.endswith(".xlsx"):
        df = pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file extension: {file_path}")
    prompt_list = df[column_name].tolist()
    print(f"Successfully loaded {len(prompt_list)} prompts.")
    return prompt_list

def get_tasks(
    prompt_list: list,
    model_version: str = "gpt-4o-2024-08-06",
    temperature: int = 0,
    logprobs: bool = True,
    top_logprobs: int = 5,
    #prompt_key: str = "{WORD}",
    company: str = "OpenAI",
    ft_dir: str = None) -> list:
    if company == "OpenAI":
        return get_tasks_openai(prompt_list, model_version, temperature, logprobs, top_logprobs)
    elif company == "Google":
        return get_tasks_gemini(prompt_list, model_version, temperature, logprobs, top_logprobs)
    elif company == "HuggingFace":
        return get_tasks_huggingface(prompt_list, model_version, temperature, logprobs, top_logprobs, ft_dir) 
    elif company == "Local":
        return get_tasks_huggingface(prompt_list, model_version, temperature, logprobs, top_logprobs, ft_dir)
    else:
        raise ValueError(f"Unknown company: {company}")

def get_tasks_openai(
    prompt_list: list,
    model_version: str = "gpt-4o-2024-08-06",
    temperature: int = 0,
    logprobs: bool = True,
    top_logprobs: int = 5,
    #prompt_key: str = "{WORD}",
) -> list:
    tasks = []
    for counter, prompt in enumerate(prompt_list, start=1):

        task = {
            "custom_id": f"{EXPERIMENT_NAME}_task_{counter}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model_version,
                "temperature": temperature,
                "logprobs": logprobs,
                "top_logprobs": top_logprobs,
                #"max_completion_tokens": 500,
                #"reasoning_effort": "minimal",
                "response_format": {"type": "text"},
                "messages": [
                {"role": "user", "content": prompt}
                ],
            },
        }

        tasks.append(task)
    return tasks

def get_tasks_gemini(
    prompt_list: list,
    model_version: str = "gemini-2.0-flash",
    temperature: float = 0.0,
    logprobs: bool = True,
    top_logprobs: int = 5,
    #prompt_key: str = "{WORD}",
) -> list:
    """
    Genera una lista de tasks con la estructura oficial de la API de Google Gemini.
    """
    tasks = []
    for counter, prompt in enumerate(prompt_list, start=1):
        task = {
            "key": f"{EXPERIMENT_NAME}_task_{counter}",
            "request": {
                "model": f"models/{model_version}",
                "contents": [
                    {
                        "parts": [
                            {
                                "text": prompt
                            }
                        ]
                    }
                ],
                "generation_config": {
                    "temperature": temperature,
                    "response_logprobs": logprobs,
                    "logprobs": top_logprobs

                }
            }
        }
        tasks.append(task)
    return tasks


def get_tasks_huggingface(
    prompt_list: list,
    model_version: str,
    temperature: float = 0.0,
    logprobs: bool = True,
    top_logprobs: int = 5,
    #prompt_key: str = "{WORD}",
    ft_dir: str = None,
) -> list:
    tasks = []
    for counter, prompt in enumerate(prompt_list, start=1):
        task = {
            "id": f"{EXPERIMENT_NAME}_task_{counter}",
            "prompt": prompt,
            "temperature": temperature,
            "response_logprobs": logprobs,
            "logprobs": top_logprobs,
            "model": model_version,
        }
        if ft_dir:
            task["ft_dir"] = ft_dir
        tasks.append(task)
    return tasks
    

def create_batches(
    tasks: list, run_prefix: str, chunk_size: int = 50000
):
    os.makedirs(f"batches", exist_ok=True)
    date_string = datetime.now().strftime("%Y-%m-%d-%H-%M")
    list_of_tasks = [
        tasks[i : i + chunk_size] for i in range(0, len(tasks), chunk_size)
    ]
    list_of_batch_names = []
    for index, tasks in enumerate(list_of_tasks):
        batch_name = f"batch_{index}_{date_string}.jsonl"
        list_of_batch_names.append(batch_name)
        with jsonlines.open(f"batches/{run_prefix}_{batch_name}", "w") as file:
                file.write_all(tasks)
    return list_of_batch_names


if __name__ == "__main__":

    if len(sys.argv) > 1:
        EXPERIMENT_NAME = sys.argv[1]
    else:
        print(
            "Provide as argument the experiment name, i.e.: python3 prepare_experiment_existing_words.py <EXPERIMENT_NAME>."
        )
        exit()


    # prepare data
    config_args = load_config(
        config_type="experiments",
        name=EXPERIMENT_NAME,
    )

    prompt_list = load_prompt_list(
        file_path=f"data/{config_args['dataset_path']}",
        column_name=config_args.get("prompt_column", "question"),
    )

    # prepare batch
    tasks = get_tasks(
        prompt_list=prompt_list,
        model_version=config_args["model_name"],
        company=config_args["company"],
        ft_dir=config_args.get("ft_dir", None),
    )

    list_of_batch_names = create_batches(
        tasks=tasks,
        run_prefix=EXPERIMENT_NAME,
    )
