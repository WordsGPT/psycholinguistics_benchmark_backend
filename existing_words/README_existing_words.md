# Existing Words

## Setting Up a New Experiment

### Initial configuration

1. Create a folder with the name `data`, add the data in XLSX format wich may include at least a column with the prompt and a column with the word to examinate.
2. Create a `config.yaml` file in the experiment folder using the content of `config_example.yaml` as a template.

### Make estimations with some model (including fine-tuning models):

1. Prepare the experiment by running `python3 prepare_experiment.py <EXPERIMENT NAME>`. -> generates the batches
2. Save the batches.jsonl files in `batches` folder
3. Run the experiment by executing `python3 execute_experiment.py <EXPERIMENT NAME>`. ->  executes the batches.

### Processing results:

1. Save the results of batches in `results` folder
2. Combine all the batches in a file called `batches.jsonl`
3. Combine all the results files in a file called `results.jsonl`

You can use these commands:

```
cat batches/*.jsonl >> batches/batches.jsonl
cat results/*.jsonl >> results/results.jsonl
```

1. Execute `python3 generateResults.py {json, number, weighted_sum} [extra-options]` for the experiment -> for your own experiment is possible that you want to modify it. We recommend creating a new file for reproducibility -> it generates a .xlsx with the results
