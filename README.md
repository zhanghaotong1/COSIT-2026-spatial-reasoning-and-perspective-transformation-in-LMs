# On Spatial Reasoning and Perspective Transformation in Language Models

## Introduction

This repository is for our work published at COSTI 2026 *On Spatial Reasoning and Perspective Transformation in Language Models*.

The `data` folder contains the training set and test set for each task. Files are in jsonl format. The `data/data_generator` folder contains all scripts to generate datasets.

The `predictions` folder contains all predicted files from the finetuned language models, grouped by different tasks. In each folder, files are named in the format *{model_name}\_{model_size}\_{random_seed}*. In the `predictions/omr` folder, we only include predicted files of DeBERTa-v3 large, because RoBERTa large and T5 base fail to converge on this task. The `predictions/gemini_pred` folder contains Gemini's predictions on main experiments.

The `main.py` script is used for the main experiments (*i.e.*, finetune a language model, test the finetuned model or test the pretrained model). The `parallel.py` script is the parallel training version of `main.py` (*i.e.*, use multiple GPUs). Use 

```
python3 main.py --help
```

or

```
python3 parallel.py --help
```

to see all possible arguments.

The `generalisation.py` script is used for the generalisation experiments. Specifically, it finetuned DeBERTa-v3 large on the simplest instances (*e.g.*, instances with complexity 3) and test on OOD instances. Use 

```
python3 generalisation.py --help
```

to see all possible arguments.

## Citation
If you use this code or data, please consider citing our paper:

```
@inproceedings{zhang2026spatial,
    title = "On Spatial Reasoning and Perspective Transformation in Language Models",
    author = "Zhang, Haotong and Pratt-Hartmann, Ian",
    booktitle = "17th International Conference on Spatial Information Theory (COSIT 2026)",
    year = "2026",
    address = "York, UK",
    pages = "7:1-7:20",
}
```
