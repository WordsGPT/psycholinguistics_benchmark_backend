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
    python prepare_experiment_ew.py <EXPERIMENT_NAME>

    EXPERIMENT_NAME:
    - <experiment_name>: process specific experiment
    - all: process all experiments
    - failed: retry failed experiments
    - status: show failed experiments

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
    company: str = "OpenAI",
    ft_dir: str = None,
    experiment_name: str = None) -> list:
    if experiment_name is None:
        experiment_name = EXPERIMENT_NAME
    
    if company == "OpenAI":
        return get_tasks_openai(prompt_list, model_version, temperature, logprobs, top_logprobs, experiment_name)
    elif company == "Google":
        return get_tasks_gemini(prompt_list, model_version, temperature, logprobs, top_logprobs, experiment_name)
    elif company == "HuggingFace":
        return get_tasks_huggingface(prompt_list, model_version, temperature, logprobs, top_logprobs, ft_dir, experiment_name) 
    elif company == "Local":
        return get_tasks_huggingface(prompt_list, model_version, temperature, logprobs, top_logprobs, ft_dir, experiment_name)
    else:
        raise ValueError(f"Unknown company: {company}")

def get_tasks_openai(
    prompt_list: list,
    model_version: str = "gpt-4o-2024-08-06",
    temperature: int = 0,
    logprobs: bool = True,
    top_logprobs: int = 5,
    experiment_name: str = None,
) -> list:
    if experiment_name is None:
        experiment_name = EXPERIMENT_NAME
        
    tasks = []
    for counter, prompt in enumerate(prompt_list, start=1):

        task = {
            "custom_id": f"{experiment_name}_task_{counter}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model_version,
                "temperature": temperature,
                "logprobs": logprobs,
                "top_logprobs": top_logprobs,
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
    experiment_name: str = None,
) -> list:
    """
    Genera una lista de tasks con la estructura oficial de la API de Google Gemini.
    """
    if experiment_name is None:
        experiment_name = EXPERIMENT_NAME
        
    tasks = []
    for counter, prompt in enumerate(prompt_list, start=1):
        task = {
            "key": f"{experiment_name}_task_{counter}",
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
    experiment_name: str = None,
) -> list:
    if experiment_name is None:
        experiment_name = EXPERIMENT_NAME
        
    tasks = []
    for counter, prompt in enumerate(prompt_list, start=1):
        task = {
            "id": f"{experiment_name}_task_{counter}",
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

def load_failed_experiments():
    """Load list of failed experiments from file"""
    failed_file = "failed_experiments/failed_prepare_exp.txt"
    try:
        if os.path.exists(failed_file):
            with open(failed_file, 'r') as f:
                failed_experiments = [line.strip() for line in f.readlines() if line.strip()]
            return failed_experiments
    except (PermissionError, OSError) as e:
        print(f"Warning: Could not read failed experiments file: {e}")
    return []

def save_failed_experiments(failed_list):
    """Save list of failed experiments to file"""
    try:
        if not failed_list:
            # If no failed experiments, remove the file and directory if empty
            failed_file = "failed_experiments/failed_prepare_exp.txt"
            if os.path.exists(failed_file):
                os.remove(failed_file)
            if os.path.exists("failed_experiments") and not os.listdir("failed_experiments"):
                os.rmdir("failed_experiments")
            return
        
        os.makedirs("failed_experiments", exist_ok=True)
        failed_file = "failed_experiments/failed_prepare_exp.txt"
        with open(failed_file, 'w') as f:
            for experiment in failed_list:
                f.write(f"{experiment}\n")
    except (PermissionError, OSError) as e:
        print(f"Warning: Could not save failed experiments file: {e}")

def remove_from_failed_experiments(experiment_name):
    """Remove a successfully executed experiment from failed list"""
    try:
        failed_experiments = load_failed_experiments()
        if experiment_name in failed_experiments:
            failed_experiments.remove(experiment_name)
            save_failed_experiments(failed_experiments)
            print(f"Removed {experiment_name} from failed experiments list")
    except Exception as e:
        print(f"Warning: Could not remove {experiment_name} from failed experiments list: {e}")

def add_to_failed_experiments(experiment_name):
    """Add a failed experiment to the failed list"""
    try:
        failed_experiments = load_failed_experiments()
        if experiment_name not in failed_experiments:
            failed_experiments.append(experiment_name)
            save_failed_experiments(failed_experiments)
            print(f"Added {experiment_name} to failed experiments list")
    except Exception as e:
        print(f"Warning: Could not add {experiment_name} to failed experiments list: {e}")

def run_single_experiment(experiment_name, all_configs):
    """Run a single experiment and return success status"""
    print(f"\n{'='*50}")
    print(f"Processing experiment: {experiment_name}")
    print(f"{'='*50}")
    
    experiment_success = False
    list_of_batch_names = []
    
    try:
        # Load configuration for this experiment
        config_args = all_configs[experiment_name]
        
        # Load prompt list
        prompt_list = load_prompt_list(
            file_path=f"data/{config_args['dataset_path']}",
            column_name=config_args.get("prompt_column", "question"),
        )

        # Prepare batch with experiment name as prefix
        tasks = get_tasks(
            prompt_list=prompt_list,
            model_version=config_args["model_name"],
            company=config_args["company"],
            ft_dir=config_args.get("ft_dir", None),
            experiment_name=experiment_name,
        )

        # Use experiment_name for batch creation
        list_of_batch_names = create_batches(
            tasks=tasks,
            run_prefix=experiment_name,
        )
        
        # Only consider successful if we have created batches
        if list_of_batch_names and len(list_of_batch_names) > 0:
            print(f"Successfully processed experiment: {experiment_name}")
            print(f"  Created batches: {list_of_batch_names}")
            experiment_success = True
        else:
            print(f"No batches created for experiment: {experiment_name}")
            experiment_success = False
        
    except Exception as e:
        print(f"Error processing experiment {experiment_name}: {str(e)}")
        experiment_success = False
    
    # Handle failed experiments list operations separately
    # This way file permission errors don't affect the experiment success status
    if experiment_success:
        remove_from_failed_experiments(experiment_name)
        return True
    else:
        add_to_failed_experiments(experiment_name)
        return False

if __name__ == "__main__":

    if len(sys.argv) > 1:
        EXPERIMENT_NAME = sys.argv[1]
    else:
        print(
            "Provide as argument the experiment name, i.e.: python3 prepare_experiment_ew.py <EXPERIMENT_NAME>."
        )
        print("Available options:")
        print("  • <experiment_name> : Run a specific experiment")
        print("  • all              : Run all experiments from config.yaml")
        print("  • failed           : Run only failed experiments from failed_prepare_exp.txt")
        print("  • status           : Check status of failed experiments")
        exit()

    # Check if user wants to see status of failed experiments
    if EXPERIMENT_NAME.lower() == "status":
        print(f"\n{'='*50}")
        print("FAILED EXPERIMENTS STATUS")
        print(f"{'='*50}")
        
        failed_experiments = load_failed_experiments()
        
        if not failed_experiments:
            print("No failed experiments found!")
            print("All experiments are running successfully.")
        else:
            print(f"Found {len(failed_experiments)} failed experiment(s):")
            print()
            for i, experiment in enumerate(failed_experiments, 1):
                print(f"{i}. {experiment}")
            
            print()
            print("To retry failed experiments:")
            print("  • Run individual experiment: python prepare_experiment_ew.py <experiment_name>")
            print("  • Run only failed experiments: python prepare_experiment_ew.py failed")
            print("  • Run all experiments (failed first): python prepare_experiment_ew.py all")
        
        print(f"{'='*50}")
        exit()

    # Check if user wants to run only failed experiments
    elif EXPERIMENT_NAME.lower() == "failed":
        # Load failed experiments
        failed_experiments = load_failed_experiments()
        
        if not failed_experiments:
            print(f"\n{'='*50}")
            print("NO FAILED EXPERIMENTS")
            print(f"{'='*50}")
            print("No failed experiments found!")
            print("All experiments are running successfully.")
            print(f"{'='*50}")
            exit()
        
        # Load all experiment configurations
        try:
            all_configs = load_config(config_type="experiments")
        except TypeError:
            try:
                full_config = load_config("experiments", name=None)
                all_configs = full_config if isinstance(full_config, dict) else {}
            except:
                import yaml
                try:
                    with open('config.yaml', 'r') as file:
                        config_data = yaml.safe_load(file)
                        all_configs = config_data.get('experiments', {})
                except FileNotFoundError:
                    print("Error: config.yaml file not found")
                    exit()
                except Exception as e:
                    print(f"Error loading config.yaml: {e}")
                    exit()
        
        if not all_configs:
            print("No experiments found in config.yaml")
            exit()
        
        print(f"\n{'='*50}")
        print("RUNNING FAILED EXPERIMENTS ONLY")
        print(f"{'='*50}")
        print(f"Found {len(failed_experiments)} failed experiments to retry:")
        for i, experiment in enumerate(failed_experiments, 1):
            print(f"  {i}. {experiment}")
        print(f"{'='*50}")
        
        successful_count = 0
        failed_count = 0
        
        for experiment_name in failed_experiments:
            if experiment_name not in all_configs:
                print(f"Warning: Experiment '{experiment_name}' not found in config.yaml, skipping...")
                # Remove it from failed experiments since it doesn't exist anymore
                remove_from_failed_experiments(experiment_name)
                continue
            
            success = run_single_experiment(experiment_name, all_configs)
            if success:
                successful_count += 1
            else:
                failed_count += 1
        
        print(f"\n{'='*50}")
        print("FAILED EXPERIMENTS EXECUTION SUMMARY")
        print(f"{'='*50}")
        print(f"Successfully processed: {successful_count}")
        print(f"Still failed: {failed_count}")
        
        # Show current failed experiments
        current_failed = load_failed_experiments()
        if current_failed:
            print(f"Experiments still in failed list: {current_failed}")
        else:
            print("All previously failed experiments now successful!")
        
        print(f"{'='*50}")
        
    # Check if user wants to run all experiments
    elif EXPERIMENT_NAME == "all":
        # Load all experiment configurations
        try:
            # Try to load config without name parameter first
            all_configs = load_config(config_type="experiments")
        except TypeError:
            # If that fails, try to load the entire config and extract experiments
            try:
                full_config = load_config("experiments", name=None)
                all_configs = full_config if isinstance(full_config, dict) else {}
            except:
                # Last resort: load config file directly
                import yaml
                try:
                    with open('config.yaml', 'r') as file:
                        config_data = yaml.safe_load(file)
                        all_configs = config_data.get('experiments', {})
                except FileNotFoundError:
                    print("Error: config.yaml file not found")
                    exit()
                except Exception as e:
                    print(f"Error loading config.yaml: {e}")
                    exit()
        
        if not all_configs:
            print("No experiments found in config.yaml")
            exit()
        
        # Load failed experiments
        failed_experiments = load_failed_experiments()
        
        # Create execution order: failed experiments first, then remaining ones (excluding those already in failed)
        remaining_experiments = [name for name in all_configs.keys() if name not in failed_experiments]
        execution_order = failed_experiments + remaining_experiments
        
        if failed_experiments:
            print(f"Found {len(failed_experiments)} previously failed experiments. Retrying them first...")
            print(f"Failed experiments: {failed_experiments}")
        
        print(f"Total experiments to process: {len(execution_order)}")
        print(f"Execution order: {execution_order}")
        
        successful_count = 0
        failed_count = 0
        
        for experiment_name in execution_order:
            if experiment_name not in all_configs:
                print(f"Warning: Experiment '{experiment_name}' not found in config.yaml, skipping...")
                # Remove it from failed experiments since it doesn't exist anymore
                if experiment_name in failed_experiments:
                    remove_from_failed_experiments(experiment_name)
                continue
            
            success = run_single_experiment(experiment_name, all_configs)
            if success:
                successful_count += 1
            else:
                failed_count += 1
        
        print(f"\n{'='*50}")
        print("EXECUTION SUMMARY")
        print(f"{'='*50}")
        print(f"Successful experiments: {successful_count}")
        print(f"Failed experiments: {failed_count}")
        
        # Show current failed experiments
        current_failed = load_failed_experiments()
        if current_failed:
            print(f"Experiments still in failed list: {current_failed}")
        else:
            print("No failed experiments remaining!")
        
        print(f"{'='*50}")
        
    else:
        # Single experiment logic
        # First check if this experiment is in the failed list
        failed_experiments = load_failed_experiments()
        
        if EXPERIMENT_NAME in failed_experiments:
            print(f"Note: {EXPERIMENT_NAME} is in the failed experiments list. Retrying from failed list...")
            
            # Load all configs to use the same function
            try:
                all_configs = load_config(config_type="experiments")
            except TypeError:
                try:
                    full_config = load_config("experiments", name=None)
                    all_configs = full_config if isinstance(full_config, dict) else {}
                except:
                    import yaml
                    try:
                        with open('config.yaml', 'r') as file:
                            config_data = yaml.safe_load(file)
                            all_configs = config_data.get('experiments', {})
                    except FileNotFoundError:
                        print("Error: config.yaml file not found")
                        exit()
                    except Exception as e:
                        print(f"Error loading config.yaml: {e}")
                        exit()
            
            if EXPERIMENT_NAME not in all_configs:
                print(f"Error: Experiment '{EXPERIMENT_NAME}' not found in config.yaml")
                remove_from_failed_experiments(EXPERIMENT_NAME)
                sys.exit(1)
            
            success = run_single_experiment(EXPERIMENT_NAME, all_configs)
            if not success:
                sys.exit(1)
        else:
            # Original single experiment logic
            config_args = load_config(
                config_type="experiments",
                name=EXPERIMENT_NAME,
            )

            success = False
            try:
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
                    experiment_name=EXPERIMENT_NAME,
                )

                list_of_batch_names = create_batches(
                    tasks=tasks,
                    run_prefix=EXPERIMENT_NAME,
                )
                
                # Only consider successful if we have created batches
                if list_of_batch_names and len(list_of_batch_names) > 0:
                    print(f"Successfully processed experiment: {EXPERIMENT_NAME}")
                    print(f"  Created batches: {list_of_batch_names}")
                    
                    # Remove from failed experiments only if truly successful
                    remove_from_failed_experiments(EXPERIMENT_NAME)
                    success = True
                else:
                    print(f"No batches created for experiment: {EXPERIMENT_NAME}")
                    add_to_failed_experiments(EXPERIMENT_NAME)
                    
            except Exception as e:
                print(f"Error processing experiment {EXPERIMENT_NAME}: {str(e)}")
                add_to_failed_experiments(EXPERIMENT_NAME)
            
            if not success:
                sys.exit(1)
