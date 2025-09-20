# Existing Words

## Setting Up a New Experiment

### Initial configuration

1. Create a folder with the name `data`, add the data in XLSX format wich may include at least a column with the prompt and a column with the word to examinate.
2. Create a `config.yaml` file in the experiment folder using the content of `config_example.yaml` as a template.

### Make estimations with some model:

1. Prepare the experiment by running `python3 prepare_experiment_existing_words.py <EXPERIMENT NAME>`. -> generates the batches
2. Save the batches files in `batches` folder (ensure that the name of the file includes the name of the experiment)
3. Run the experiment by executing `python3 execute_experiment_existing_words.py <EXPERIMENT NAME>`. ->  executes the batches.

### Processing results:

1. Save the results of batches in `results` folder (ensure that the name of the file includes the name of the experiment)
2. Execute `python3 generateResults_existing_words.py <EXPERIMENT NAME> {json, number, weighted_sum} [extra-options]` -> it generates a .xlsx with the results of the experiment in the folder `outputs/<EXPERIMENT NAME>`
