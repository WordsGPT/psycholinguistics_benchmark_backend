# Existing Words

## Setting Up a New Experiment

### Initial configuration

1. Create a folder with the name `data`, add the data in XLSX format wich may include at least a column with the prompt and a column with the word to examinate.
2. Create a `config.yaml` file in the experiment folder using the content of `config_example.yaml` as a template.

### Make estimations with some model:

1. Prepare the experiment by running `python3 prepare_experiment_ew.py <EXPERIMENT NAME>`. -> generates the batches**
      **EXPERIMENT_NAME:**
         * <experiment_name>: process specific experiment.
         * "all": process all experiments.
         * "failed": retry failed experiments.
         * "status": show failed experiments.

2. Save the batches files in `batches` folder (ensure that the name of the file includes the name of the experiment)

3. Run the experiment by executing `python3 execute_experiment_ew.py <EXPERIMENT NAME>`. ->  executes the batches.EXPERIMENT_NAME:
      **EXPERIMENT_NAME:**
         * <experiment_name>: process specific experiment.
         * "all": process all experiments in the batches folder.
         * "remain": check and download batches still in tracking.
         * "status": show batches still in tracking.

### Processing results:

1. Save the results of batches in `results` folder (ensure that the name of the file includes the name of the experiment)
2. Execute `python3 generateResults_ew.py <EXPERIMENT NAME> {json, number, weighted_sum} [extra-options]` -> it generates a .xlsx with the results of the experiment in the folder `outputs/<EXPERIMENT NAME>`
      **EXPERIMENT_NAME:**
         * <experiment_name>: process specific experiment. Need to select a mode.
         * "all": process all experiments. Need to select a mode.
         * "failed": retry failed experiments. Do not select a mode.
         * "status": show failed experiments. Do not select a mode.

