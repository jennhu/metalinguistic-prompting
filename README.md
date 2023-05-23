# Prompt-based methods may underestimate large language models' linguistic generalizations

This repository contains materials for the paper
"Prompt-based methods may underestimate large language models' linguistic generalizations" 
(Hu & Levy, 2023). The preprint is available on both [arXiv](https://arxiv.org/abs/2305.13264)
and [LingBuzz](https://lingbuzz.net/lingbuzz/007313).

If you find the code or data useful in your research, please use the following citation:

```
@document{hu_prompt-based_2023,
    title = {Prompt-based methods may underestimate large language models' linguistic generalizations},
    author = {Hu, Jennifer and Levy, Roger},
    year = {2023},
    url = {https://lingbuzz.net/lingbuzz/007313}
}
```

## Evaluation materials

Evaluation datasets can be found in the [`datasets`](datasets) folder.
Please refer to the README in that folder for more details on how the stimuli were assembled and formatted.

## Evaluation scripts

The [`scripts`](scripts) folder contains scripts for running the experiments. 
There are separate scripts for models accessed through Huggingface (`*hf.sh`) and the OpenAI API (`*openai.sh`).

For example, to evaluate `flan-t5-small` on the SyntaxGym dataset of Experiment 3b, run the following command 
from the root of this directory:
```
bash scripts/run_exp3b_hf.sh syntaxgym google/flan-t5-small flan-t5-small
```

Please note that to run the OpenAI models, you will need to save your OpenAI API key to a file named `key.txt`
in the root of this directory. For security reasons, **do not** commit this file (it is ignored in `.gitignore`).


## Results and analyses

The results from the paper can be accessed by extracting the [`results.zip`](results.zip) file.
This will create a folder called `results`, which is organized by experiment:
- `exp1_word-prediction`
- `exp2_word-comparison`
- `exp3a_sentence-judge`
- `exp3b_sentence-comparison`

A few notes about the results:
- Each result file is named in the following format: `<dataset>_<model>_<eval_type>.json`.
  For the experiments where option order matters, there is an addition `_<option_order>` suffix in the name.
- Each result file is formatted as a JSON file, with two dictionaries:
    - `meta` contains meta information about the run (e.g., name of model, timestamp of run, path to data file)
    - `results` contains the results from the run, formatted as a list of dictionaries (one per stimulus item)
- The results from the `direct` evaluation method are identical across Experiments 3a and 3b (see paper for details).

The figures from our paper can be reproduced using the [`analysis.ipynb`](analysis.ipynb) notebook.
