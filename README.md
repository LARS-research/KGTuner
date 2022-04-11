# KGTuner
Official code for the paper "KGTuner: Efficient Hyper-parameter Search for Knowledge Graph Learning" (ACL 2022 long paper).

[TOC]

## Overview

<img src="./misc/KGTuner.pdf" alt="KGTuner" style="zoom: 67%;" />

We propose an efficient two-stage search algorithm, KGTuner, which efficiently explores HP configurations on small subgraph at the first stage and transfers the top-performed configurations for fine-tuning on the large full graph at the second stage.



## Instructions

### 0. Setup
```
pip install requirements.txt
```

### 1. Generate sampled KG
Supported datasets: `DATASET = {wn18rr/FB15k_237/ogbl-biokg/ogbl-wikikg2}`.

usage:
```
python3 generator_random_walks.py -dataset {DATASET} -sample_ratio {FLOAT}
```
example:
```
python3 generator_random_walks.py -dataset wn18rr -sample_ratio 0.2
```
The generated new dataset will be store in `dataset/sampled_wn18rr_0.2_starts_10`.

### 2. Run hyper-parameter searching
Supported models: `MODEL = {ComplEx/DistMult/TransE/RotatE/TuckER/RESCAL/ConvE/AutoSF}`.

usage:

```
python3 run.py -search -earlyStop -space {SPACE} -dataset {DATASET} -cpu 2 -valid_steps {e.g., 2500} -max_steps {e.g., 100000} -model {MODEL} -eval_test -test_batch_size {e.g., 16} -gpu 0
```

#### Two-stage searching
example (firstly search on `sampled_wn18rr_0.2_starts_10` and then on `wn18rr`):
```
stage1:
python3 run.py -search -earlyStop -space reduced -dataset sampled_wn18rr_0.2_starts_10 -cpu 2 -valid_steps 2500 -max_steps 20000 -model ComplEx -eval_test -test_batch_size 16 -gpu 0

stage2:
python3 run.py -search -earlyStop -space full -dataset wn18rr -pretrain_dataset sampled_wn18rr_0.2_starts_10 -cpu 2 -valid_steps 2500 -max_steps 100000 -model ComplEx -eval_test -test_batch_size 16 -gpu 0
```

#### One-stage searching

Alternatively, you can proceed searching directly on the original KG.

example:
```
python3 run.py -search -earlyStop -space full -dataset wn18rr -cpu 2 -valid_steps 2500 -max_steps 100000 -model ComplEx -eval_test -test_batch_size 16 -gpu 0
```

### 3. Show searched results
You can use the `showResults.py` to check your local HPO trials.

usage:
```
python3 showResults.py -dataset {DATASET} -model {MODEL}
```
example:
```
python3 showResults.py -dataset wn18rr -model ComplEx
```



## Results

WN18RR

| Model    | Test MRR | Test Hit@1 | Test Hit@3 | Test Hit@10 |
| -------- | -------- | ---------- | ---------- | ----------- |
| ComplEx  | 0.484    | 0.440      | 0.506      | 0.562       |
| DistMult | 0.453    | 0.407      | 0.468      | 0.548       |
| RESCAL   | 0.479    | 0.436      | 0.496      | 0.557       |
| ConvE    | 0.437    | 0.399      | 0.449      | 0.515       |
| TransE   | 0.233    | 0.032      | 0.399      | 0.542       |
| RotatE   | 0.480    | 0.427      | 0.501      | 0.582       |
| TuckER   | 0.480    | 0.437      | 0.500      | 0.557       |

FB15k-237

| Model    | Test MRR | Test Hit@1 | Test Hit@3 | Test Hit@10 |
| -------- | -------- | ---------- | ---------- | ----------- |
| ComplEx  | 0.352    | 0.263      | 0.387      | 0.530       |
| DistMult | 0.345    | 0.254      | 0.377      | 0.527       |
| RESCAL   | 0.357    | 0.268      | 0.390      | 0.535       |
| ConvE    | 0.335    | 0.242      | 0.368      | 0.523       |
| TransE   | 0.327    | 0.228      | 0.369      | 0.522       |
| RotatE   | 0.338    | 0.243      | 0.373      | 0.527       |
| TuckER   | 0.347    | 0.255      | 0.382      | 0.534       |

ogbl-biokg

| Model    | Test MRR      | Val MRR       | #parameters |
| -------- | ------------- | ------------- | ----------- |
| ComplEx  | 0.8385±0.0009 | 0.8394±0.0007 | 187,648,000 |
| DistMult | 0.8241±0.0008 | 0.8245±0.0009 | 93,824,000  |
| RotatE   | 0.8013±0.0015 | 0.8024±0.0012 | 187,597,000 |
| TransE   | 0.7781±0.0009 | 0.7787±0.0008 | 187,648,000 |
| AutoSF   | 0.8354±0.0013 | 0.8361±0.0012 | 187,648,000 |

ogbl-wikikg2

| Model    | Test MRR      | Val MRR       | #parameters |
| -------- | ------------- | ------------- | ----------- |
| ComplEx  | 0.4942±0.0017 | 0.5099±0.0023 | 250,113,900 |
| DistMult | 0.4837±0.0078 | 0.5004±0.0075 | 250,113,900 |
| RotatE   | 0.2948±0.0026 | 0.2650±0.0034 | 250,087,150 |
| TransE   | 0.4739±0.0021 | 0.4932±0.0013 | 250,113,900 |
| AutoSF   | 0.5222±0.0021 | 0.5397±0.0023 | 250,113,900 |



## Reproduction

Detailed commands can be found in `code/scripts/reproduce.sh`.



## Citation

If you find this repository useful in your research, please kindly cite our paper.
```
The bib is incoming.
```

