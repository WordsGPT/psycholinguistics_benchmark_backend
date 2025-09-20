# Existing Words

## Setting Up a New Experiment

### Initial configuration

1. Create a folder with the name `data`, add the data in XLSX format wich may include at least a column with the prompt and a column with the word to examinate.
2. Create a `config.yaml` file in the experiment folder using the content of `config_example.yaml` as a template.

### Make estimations with some model:

1. Prepare the experiment by running `python3 prepare_experiment_ew.py <EXPERIMENT NAME>`. -> generates the batches
2. Save the batches files in `batches` folder (ensure that the name of the file includes the name of the experiment)
3. Run the experiment by executing `python3 execute_experiment_ew.py <EXPERIMENT NAME>`. ->  executes the batches.

### Processing results:

1. Save the results of batches in `results` folder (ensure that the name of the file includes the name of the experiment)
2. Execute `python3 generateResults_ew.py <EXPERIMENT NAME> {json, number, weighted_sum} [extra-options]` -> it generates a .xlsx with the results of the experiment in the folder `outputs/<EXPERIMENT NAME>`

## Automated experiment execution (`run_scripts_ew.py`)

Runs all experiments defined in `config.yaml` using a specified script, and automatically retries only failed experiments on subsequent runs.

### Usage

Execute `python3 run_scripts_ew.py <SCRIPT_NAME>` -> `<SCRIPT_NAME>` :The script to run for each experiment.

### How it works

1. Runs all experiments listed in `config.yaml`, or only those that failed in the previous run.
2. Failed experiments are saved in `failed_exp/failed_<SCRIPT_NAME>_experiments.txt`.
3. If all experiments succeed, the failed experiments file and folder are deleted automatically.
4. Displays the number of successful experiments and lists any that failed.
