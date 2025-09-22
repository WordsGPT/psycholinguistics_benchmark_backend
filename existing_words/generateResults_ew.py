""" USAGE:
python generateResults__ew.py <EXPERIMENT_NAME> [mode] [language]

EXPERIMENT_NAME:
- <experiment_name>: process specific experiment. Need to select a mode.
- all: process all experiments. Need to select a mode.
- failed: retry failed experiments. Not select a mode.
- status: show failed experiments status. Not select a mode.

modes: 
- json (output of estimations is a JSON with the word and its prediction. It checks if the word in the input matches the word in the output)
- weighted_sum (wheighted sum of the logprobs of the tokens in the word. Only valid if single token output)
- number (output of estimations is a number, the estimation of the word)

languages:
- german
"""



import json
import csv
import os
import glob
from pathlib import Path
import numpy as np
import re
import time
import pandas as pd
import sys

# Add utils import
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import read_yaml

def extract_word_input(text):
    match = re.findall(f'{open_quotations_constant}(.*?){closing_quotations_constant}', text)
    if match and len(match) >= 3:
        return match[2]
    return None


def extract_word_output(text):
    match = re.search(rf'"{word_constant}"\s*:\s*"([^"]+)"', text)
    if match:
        word = match.group(1)
        return word
    return None


def extract_number(text):
    match = re.search(rf'"{feature_constant}"\s*:\s*"([0-9]*\.?[0-9]+)"', text)
    if not match:
        match = re.search(rf'"{feature_constant}"\s*:\s*([0-9]*\.?[0-9]+)', text)
    if not match:
        all_matches = re.findall(r'[-+]?\d*\.\d+|[-+]?\d*\d+\.\d*|[-+]?\d+', text)
        if all_matches:
            return float(all_matches[-1])
    if match:
        return float(match.group(1))
    return None


def read_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

## OpenAI ##

def openAI_processing(results_content_file, batches_content_file):
    match_key = 'custom_id'
    lookup = {entry[match_key]: entry for entry in batches_content_file if match_key in entry}
    combined_data = []
    index = 0
    for entry in results_content_file:
        entry_result = {}
        index += 1
        weighted_sum = None
        logprob = None
        if match_key in entry and entry[match_key] in lookup:
            combined_entry = {**entry, **lookup[entry[match_key]]}
            custom_id = combined_entry['custom_id']
            if mode == "json":             
                word_input = extract_word_input(combined_entry['body']['messages'][0]['content'])
                word_output = extract_word_output(combined_entry['response']['body']['choices'][0]['message']['content'])

                feature_value = extract_number(combined_entry['response']['body']['choices'][0]['message']['content'])
                if word_input and word_output:
                    if word_input != word_output:
                        print(f"Warning: custom Id: '{custom_id}. Word input '{word_input}' does not match word output '{word_output}'")
                        #feature_value = '#N/D'
            elif mode == "weighted_sum" or mode == "number":
                word_input = extract_word_input(combined_entry['body']['messages'][0]['content'])
                # Only valid for responses of single token
                if len(combined_entry["response"]["body"]["choices"][0]["logprobs"]["content"]) == 1:
                    top_logprobs_list = combined_entry["response"]["body"]["choices"][0]["logprobs"]["content"][0]['top_logprobs']
                    weighted_sum = 0
                    # Iterate over the list of top_logprobs that are numbers
                    for top_logprob in top_logprobs_list:
                        try:
                            token_value = int(top_logprob['token'])
                            logprob_value = top_logprob['logprob']
                            weighted_sum += token_value * np.exp(float(logprob_value))
                        except ValueError:
                            pass
                    logprob = combined_entry['response']['body']['choices'][0]['logprobs']['content'][0]['logprob']
                feature_value = combined_entry['response']['body']['choices'][0]['message']['content']

            entry_result['word'] = word_input
            # entry_result['custom_id'] = custom_id
            entry_result[feature_column] = feature_value

            if logprob is not None:
                entry_result['logprob'] = logprob
                logprobs = True
            if weighted_sum is not None:
                entry_result['weighted_sum'] = weighted_sum

            combined_data.append(entry_result)

    return combined_data

## Google ##

def google_processing(results_content_file, batches_content_file):
    """Deserialize Google (Gemini) batch/results JSONL and return rows.

    Expects structure like:
    - batches *.jsonl lines: {"key": str, "request": {"contents": [{"parts": [{"text": str}, ...]}], ...}}
    - results *.jsonl lines: {"key": str, "response": {"candidates": [{"content": {"parts": [{"text": str}]}, "logprobsResult": {...}}]}}
    """
    # Build lookup from batches by key/custom_id (prefer 'key' for Google format)
    def get_match_value(obj):
        return obj.get('key')
    
    lookup = {}
    for entry in batches_content_file:
        matck_key = get_match_value(entry)
        if matck_key is not None:
            lookup[matck_key] = entry

    combined_data = []
    for entry in results_content_file:
        matck_key = get_match_value(entry)
        if matck_key is None or matck_key not in lookup:
            continue

        batch_item = lookup[matck_key]
        result_item = entry.get('response', {})

        # Extract the prompt text (to recover the word inside quotes)
        prompt_text = None
        try:
            contents = batch_item.get('request', {}).get('contents', [])
            parts_texts = []
            for c in contents:
                for p in c.get('parts', []):
                    t = p.get('text')
                    if isinstance(t, str):
                        parts_texts.append(t)
            if parts_texts:
                prompt_text = "\n".join(parts_texts)
        except Exception:
            prompt_text = None

        word_input = extract_word_input(prompt_text) if prompt_text else None

        # Extract the model output text
        output_text = None
        try:
            candidates = result_item.get('candidates', [])
            if candidates:
                content = candidates[0].get('content', {})
                parts = content.get('parts', [])
                if parts:
                    output_text = parts[0].get('text')
        except Exception:
            output_text = None

        # Parse number from output text
        feature_value = extract_number(output_text or '')

        entry_result = {
            'word': word_input,
            feature_column: feature_value
        }

        # Optional: compute weighted_sum and logprob for single numeric token outputs
        try:
            if mode in ("weighted_sum", "number"):
                is_token_content_single_token_number = str(output_text).isdigit() and int(output_text) < 1000 # only positive integers and tokenizers until 999 with one token
                candidates = result_item.get('candidates', [])
                if candidates:
                    cand0 = candidates[0]
                    lpr = cand0.get('logprobsResult') or {}
                    chosen = lpr.get('chosenCandidates') or []
                    if chosen:
                        # Logprob of the chosen first token (likely the digit)
                        entry_result['logprob'] = chosen[0].get('logProbability')
                    top_list = lpr.get('topCandidates') or []

                    if is_token_content_single_token_number: 
                        # Only use the first step (the numeric token) for weighted sum
                        top_first = top_list[0]
                        weighted_sum = 0.0
                        for tc in top_first.get('candidates', []):
                            tok = tc.get('token')
                            lp = tc.get('logProbability')
                            try:
                                tok_val = int(tok)
                                weighted_sum += tok_val * float(np.exp(float(lp)))
                            except Exception:
                                pass
                        entry_result['weighted_sum'] = weighted_sum
        except Exception:
            # Make logprob/weighted_sum optional; ignore failures silently
            pass

        combined_data.append(entry_result)

    return combined_data


## HuggingFace ##

def huggingface_processing(results_content_file, batches_content_file):
    match_key = 'id'
    lookup = {entry[match_key]: entry for entry in batches_content_file if match_key in entry}
    combined_data = []
    index = 0
    for entry in results_content_file:
        entry_result = {}
        index += 1
        weighted_sum = None
        logprob = None
        if match_key in entry and entry[match_key] in lookup:
            combined_entry = {**entry, **lookup[entry[match_key]]}
            custom_id = combined_entry[match_key]
            if mode == "json":             
                word_input = extract_word_input(combined_entry['prompt'])
                word_output = extract_word_output(combined_entry['response']['body']['choices'][0]['message']['content'])

                feature_value = extract_number(combined_entry['response']['body']['choices'][0]['message']['content'])
                if word_input and word_output:
                    if word_input != word_output:
                        print(f"Warning: custom Id: '{custom_id}. Word input '{word_input}' does not match word output '{word_output}'")
                        #feature_value = '#N/D'
            elif mode == "weighted_sum" or mode == "number":
                word_input = extract_word_input(combined_entry['prompt'])
                feature_value = combined_entry['response']['body']['choices'][0]['message']['content']
                is_token_content_single_token_number = str(feature_value).isdigit() and int(feature_value) < 1000
                # Only valid for responses of single util tokens
                if is_token_content_single_token_number:
                    top_logprobs_list = combined_entry["response"]["body"]["choices"][0]["logprobs"]["content"][0]['top_logprobs']
                    weighted_sum = 0
                    # Iterate over the list of top_logprobs that are numbers
                    for top_logprob in top_logprobs_list:
                        try:
                            token_value = int(top_logprob['token'])
                            logprob_value = top_logprob['logprob']
                            weighted_sum += token_value * np.exp(float(logprob_value))
                        except ValueError:
                            pass
                    logprob = combined_entry['response']['body']['choices'][0]['logprobs']['content'][0]['logprob']
                

            entry_result['word'] = word_input
            # entry_result['custom_id'] = custom_id
            entry_result[feature_column] = feature_value

            if logprob is not None:
                entry_result['logprob'] = logprob
                logprobs = True
            if weighted_sum is not None:
                entry_result['weighted_sum'] = weighted_sum

            combined_data.append(entry_result)

    return combined_data

## Failed experiments management ##

def add_failed_experiment(experiment_name, experiment_mode):
    """Add a failed experiment to the failed experiments file."""
    os.makedirs("failed_experiments", exist_ok=True)
    failed_file = "failed_experiments/failed_generateResults.xlsx"
    
    # Create new entry
    new_entry = {"experiment_name": experiment_name, "mode": experiment_mode}
    
    # Check if file exists and read existing data
    if os.path.exists(failed_file):
        try:
            df = pd.read_excel(failed_file)
            # Check if this experiment+mode combination already exists
            if not ((df['experiment_name'] == experiment_name) & (df['mode'] == experiment_mode)).any():
                df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
        except Exception:
            df = pd.DataFrame([new_entry])
    else:
        df = pd.DataFrame([new_entry])
    
    df.to_excel(failed_file, index=False)
    print(f"Added failed experiment: {experiment_name} (mode: {experiment_mode})")

def remove_failed_experiment(experiment_name, experiment_mode):
    """Remove a successfully processed experiment from the failed experiments file."""
    failed_file = "failed_experiments/failed_generateResults.xlsx"
    
    if not os.path.exists(failed_file):
        return
    
    try:
        df = pd.read_excel(failed_file)
        # Remove the specific experiment+mode combination
        df = df[~((df['experiment_name'] == experiment_name) & (df['mode'] == experiment_mode))]
        
        if df.empty:
            # Delete file if no failed experiments remain
            os.remove(failed_file)
            print(f"No failed experiments remaining. Removed {failed_file}")
        else:
            df.to_excel(failed_file, index=False)
            print(f"Removed successful experiment: {experiment_name} (mode: {experiment_mode})")
    except Exception as e:
        print(f"Error updating failed experiments file: {e}")

def get_failed_experiments():
    """Get list of failed experiments."""
    failed_file = "failed_experiments/failed_generateResults.xlsx"
    
    if not os.path.exists(failed_file):
        return []
    
    try:
        df = pd.read_excel(failed_file)
        return [(row['experiment_name'], row['mode']) for _, row in df.iterrows()]
    except Exception:
        return []

def show_failed_experiments_status():
    """Show status of failed experiments."""
    failed_experiments = get_failed_experiments()
    
    if not failed_experiments:
        print("No failed experiments found.")
    else:
        print(f"Number of failed experiments: {len(failed_experiments)}")
        print("Failed experiments:")
        for exp_name, exp_mode in failed_experiments:
            print(f"  - {exp_name} (mode: {exp_mode})")

def get_all_experiments():
    """Get all unique experiment names from config.yaml that have both batch and result files."""
    try:
        # Load experiments from config.yaml like run_scripts_ew.py does
        config = read_yaml("config.yaml")
        potential_experiments = list(config.get("experiments", {}).keys())
        print(f"Found {len(potential_experiments)} experiments in config: {potential_experiments}")
    except Exception as e:
        print(f"Error reading config.yaml: {e}")
        return set()
    
    # Check which experiments have both batch and result files
    valid_experiments = set()
    
    for exp_name in potential_experiments:
        matches_batches = glob.glob(f"batches/*{exp_name}*.jsonl")
        matches_results = glob.glob(f"results/*{exp_name}*.jsonl")
        
        print(f"Testing experiment '{exp_name}':")
        print(f"  - Batch matches: {matches_batches}")
        print(f"  - Result matches: {matches_results}")
        
        if matches_batches and matches_results:
            valid_experiments.add(exp_name)
            print(f"  Valid experiment found: {exp_name}")
        else:
            print(f"  Invalid: missing files - batches: {len(matches_batches)}, results: {len(matches_results)}")
    
    print(f"Total valid experiments: {len(valid_experiments)}")
    return valid_experiments

def get_all_experiments_from_config():
    """Get all experiment names from config.yaml regardless of file availability."""
    try:
        config = read_yaml("config.yaml")
        experiments = list(config.get("experiments", {}).keys())
        print(f"All experiments from config: {experiments}")
        return experiments
    except Exception as e:
        print(f"Error reading config.yaml: {e}")
        return []

def process_single_experiment(experiment_name, experiment_mode, country=""):
    """Process a single experiment and return success status."""
    global mode, open_quotations_constant, closing_quotations_constant
    global word_constant, feature_column, feature_constant, logprobs
    
    mode = experiment_mode
    
    # Check if this experiment is already in failed list to avoid duplication
    failed_experiments = get_failed_experiments()
    is_in_failed_list = any(exp[0] == experiment_name and exp[1] == experiment_mode 
                           for exp in failed_experiments)
    
    try:
        # SELECT THE BATCH AND RESULT FILES OF THE EXPERIMENT #
        matches_batches = glob.glob(f"batches/*{experiment_name}*.jsonl")
        matches_results = glob.glob(f"results/*{experiment_name}*.jsonl")

        if not matches_batches:
            if not is_in_failed_list:
                add_failed_experiment(experiment_name, experiment_mode)
            print(f"No batches file found for experiment name '{experiment_name}' in 'batches' folder.")
            return False

        if not matches_results:
            if not is_in_failed_list:
                add_failed_experiment(experiment_name, experiment_mode)
            print(f"No results file found for experiment name '{experiment_name}' in 'results' folder.")
            return False

        batches_file = matches_batches[0]
        results_file = matches_results[0]

        # CREATE OUTPUTS FOLDER IF IT DOESN'T EXIST#
        os.makedirs(f"outputs/{experiment_name}", exist_ok=True)

        # Initialize constants
        open_quotations_constant = '"'
        closing_quotations_constant = '"'
        word_constant = 'Word'
        feature_column = 'familiarity'
        feature_constant = 'AoA'
        logprobs = False
        timestamp = int(time.time())
        output_file = f'outputs/{experiment_name}/output_{experiment_name}_{mode}_{timestamp}.xlsx'

        if country == 'german':
            word_constant = 'Wort'
            feature_constant = 'Erwerbsalter'
            open_quotations_constant = '„'
            closing_quotations_constant = '"'

        results_content_file = read_jsonl(results_file)
        batches_content_file = read_jsonl(batches_file)

        if 'custom_id' in batches_content_file[0]:
            combined_data = openAI_processing(results_content_file, batches_content_file)
        elif 'key' in batches_content_file[0]:
            combined_data = google_processing(results_content_file, batches_content_file)
        elif 'id' in batches_content_file[0]:
            combined_data = huggingface_processing(results_content_file, batches_content_file)
        else:
            if not is_in_failed_list:
                add_failed_experiment(experiment_name, experiment_mode)
            print("Unknown batch format, cannot process results.")
            return False

        all_fieldnames = list(combined_data[0].keys())

        df = pd.DataFrame(combined_data)
        df.to_excel(output_file, index=False, columns=all_fieldnames)

        print(f"Combined data written to {output_file}")
        
        # Remove from failed experiments if it was there
        if is_in_failed_list:
            remove_failed_experiment(experiment_name, experiment_mode)
        
        return True

    except Exception as e:
        if not is_in_failed_list:
            add_failed_experiment(experiment_name, experiment_mode)
        print(f"Error processing experiment '{experiment_name}': {e}")
        return False

## Main ##

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Provide as arguments the experiment name [or 'all', 'failed', 'status'], the mode [json, number, weighted_sum] and optionally the language.")
        exit()

    EXPERIMENT_NAME = sys.argv[1]
    
    # Handle special commands
    if EXPERIMENT_NAME == "status":
        show_failed_experiments_status()
        exit()
    
    if EXPERIMENT_NAME == "failed":
        failed_experiments = get_failed_experiments()
        if not failed_experiments:
            print("No failed experiments to retry.")
            exit()
        
        print(f"Retrying {len(failed_experiments)} failed experiments...")
        for exp_name, exp_mode in failed_experiments:
            print(f"Retrying experiment: {exp_name} (mode: {exp_mode})")
            process_single_experiment(exp_name, exp_mode)
        exit()
    
    # Get mode and country for normal processing
    if len(sys.argv) < 3:
        print("Provide as arguments the experiment name [or 'all', 'failed', 'status'], the mode [json, number, weighted_sum] and optionally the language.")
        exit()
    
    mode = sys.argv[2]
    country = sys.argv[3] if len(sys.argv) > 3 else ""
    
    if EXPERIMENT_NAME == "all":
        # Get ALL experiments from config.yaml (not just those with files)
        all_experiments_from_config = get_all_experiments_from_config()
        
        if not all_experiments_from_config:
            print("No experiments found in config.yaml.")
            exit()
        
        print(f"Processing {len(all_experiments_from_config)} experiments from config in mode '{mode}'...")
        successful = 0
        failed = 0
        
        for exp_name in all_experiments_from_config:
            print(f"\nProcessing experiment: {exp_name}")
            if process_single_experiment(exp_name, mode, country):
                successful += 1
            else:
                failed += 1
        
        print(f"\nProcessing complete: {successful} successful, {failed} failed")
    else:
        # Process single experiment
        process_single_experiment(EXPERIMENT_NAME, mode, country)

