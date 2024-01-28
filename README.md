# gec-t5

A reproduction of training T5 on cLang-8 (corresponding to Table 4) in the following paper:
```
@inproceedings{rothe-etal-2021-simple,
    title = "A Simple Recipe for Multilingual Grammatical Error Correction",
    author = "Rothe, Sascha  and
      Mallinson, Jonathan  and
      Malmi, Eric  and
      Krause, Sebastian  and
      Severyn, Aliaksei",
    editor = "Zong, Chengqing  and
      Xia, Fei  and
      Li, Wenjie  and
      Navigli, Roberto",
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 2: Short Papers)",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.acl-short.89",
    doi = "10.18653/v1/2021.acl-short.89",
    pages = "702--707"
}
```

Confirmed that it works on python 3.11.0.

# Installation

```sh
# python -m venv env
# source env/bin/activate
pip install -r requirements.txt
wget https://github.com/google-research-datasets/clang8/raw/main/retokenize.py
```

# Procedure

### 1. Train a model
The follwing example trains a model on four GPUs.  
- For the daataset path, `{train|valid}pref + '.' + {source|target}` is used.
- E.g. `path/to/train.src` and `path/to/train.trg` are used for the training data in the below example.
Note that the script uses Accelerate module, so you need to use `accelerate lanuch` instead of `python`.
```sh
accelerate launch \
--multi_gpu \
--num_processes 4 \
train.py \
    --trainpref path/to/train \
    --validpref path/to/valid \
    --source src \
    --target trg \
    --max_len 128 \
    --seed 10 \
    --batch_size 64 \
    --accumulation 2 \
    --epochs 5 \
    --outdir outputs/sample/
```

The format of the output directory is like this:
```
outputs/sample/
└── seed10
    ├── best
    │   ├── added_tokens.json
    │   ├── config.json
    │   ├── generation_config.json
    │   ├── merges.txt
    │   ├── model.safetensors
    │   ├── special_tokens_map.json
    │   ├── tokenizer_config.json
    │   ├── tokenizer.json
    │   ├── training_state.json
    │   └── vocab.json
    ├── last
    │   ├── The same as best/
    └── log.json
```

### 2. Inference

`generate.py` can be used for inference.  
This script processes the input sentences in the sorted order by length, for faster inference.  
The output is be shown in your terminal, so use redirection to save it to a file.
```
python generate.py \
    --input <a raw text file> \
    --restore_dir <path to the directory, like outputs/sample/best in the above example> \
    --batch_size 128 \
    > output.txt
```

# Reproduction experiments

I performed experiments with this implementation. 

### Prepare cLang-8 dataset
Follow [the official instruction](https://github.com/google-research-datasets/clang8?tab=readme-ov-file#dataset-preparation).

### Training

I fine-tuned `google/t5-v1_1-XXX` models.

Common settings are the following.
|Param|Value|
|:--|:--|
|seed|10|
|lr|1e-5|
|lr scheduler|linear|
|warmup|10000 steps|
|Optimizer|AdamW|
|max length|128 (both encoder and decoder)|
|mixed precision|bf16|
|training data|cLang-8 (en, 2,372,119 sents)|
|validation data|BEA19-dev (W&I+L-dev, 4,384 sents)|
|test data|CoNLL2014, BEA19-test|

Some settings are different between model size:
- Small
```
GPU: four RTX3090s (about 50 min. per epoch)
batch size: 64
gradients accumulation 2
epochs: 100 (the best is at 91 epoch)
```

- Base
```
GPU: four RTX3090s (about 150 min. per epoch)
batch size: 64
gradients accumulation 2
epochs: 10
```

- Large
```
GPU: four A100s (about 160 min. per epoch)
batch size: 32
gradients accumulation 4
epochs: 10
```

### Evaluation
The checkpoint that achieves minimum loss on BEA19-dev was used for the evaluation.
- For CoNLL-2014, I re-tokenized the output sentences by the following. Then M2scorer was used to evaluate.
    ```sh
    python retokenize.py < conll14.out > conll14_retok.out
    ```
- For BEA19-dev, I re-extract correction spans of the official reference by the following (see Sec4.1 in [Bryant+ 17](https://aclanthology.org/P17-1074/)). Then ERRANT was used to evaluate.
    ```sh
    errant_m2 -auto path/to/wi+locness/m2/ABCN.dev.gold.bea19.m2 -out path/to/new.m2
    ```
- For BEA19-test, I used [CodaLab's open phase submission](https://codalab.lisn.upsaclay.fr/competitions/4057).

### Results
The pre-trained models are available from Hugging Face Hub. You can use these models by specifying `--restore_dir` of `generate.py`.

```
python generate.py \
    --input <a raw text file> \
    --restore_dir gotutiyan/gec-t5-base-clang8 \
    --batch_size 64 \
    > <path to output file>
```

|Model|CoNLL14 (P/R/F0.5)|BEA19-dev (P/R/F0.5)|BEA19-test (P/R/F0.5)|
|:--|:-:|:-:|:-:|
|paper (t5-small)|-/-/60.54|-|-/-/65.01|
|paper (t5-base)|-/-/65.05|-|-/-/69.38|
|paper (t5-large)|-/-/66.04|-|-/-/72.06|
|paper (t5-xl)|-/-/67.65|-|-/-/73.92|
|paper (t5-xxl)|-/-/68.75|-|-/-/75.88|
|[gotutiyan/gec-t5-small-clang8](https://huggingface.co/gotutiyan/gec-t5-small-clang8)|68.96 / 41.17 / 60.76|60.34 / 34.00 / 52.24|71.47 / 54.14 / 67.17|
|[gotutiyan/gec-t5-base-clang8](https://huggingface.co/gotutiyan/gec-t5-base-clang8)|72.98 / 45.24 / 65.01|63.65 / 37.01 / 55.64|74.15 / 58.28 / 70.32|
|[gotutiyan/gec-t5-large-clang8](https://huggingface.co/gotutiyan/gec-t5-large-clang8)|74.48 / 48.63 / 67.32|65.94 / 40.64 / 58.64|76.88 / 62.27 / 73.44|
|gotutiyan/gec-t5-xl-clang8|TBA?|||
|gotutiyan/gec-t5-xxl-clang8|TBA?|||

### Insights
- In CoNLL-2014, `retokenize.py` improves F0.5 more than 2 points.
- For `google/t5-v1_1-small`, at least 10 epochs could not reproduce the competetive results of the paper. The above results are checkpoints at 90 epochs. This means that even after 90 epochs of training, the minimum loss is still achieved with BEA19-dev. (Probably, increasing the learning rate is better)
    - 10 epochs seems enough for `google/t5-v1_1-base` and `google/t5-v1_1-large`.

# License

The same as cLang-8 corpus and the original Lang-8 corpus, the pre-trained models are distributed for research and educational purposes only. Specifically, they are released under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) license.

The code is distributed under MIT license.