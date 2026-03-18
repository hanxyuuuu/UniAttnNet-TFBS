# UniAttnNet-TFBS

UniAttnNet-TFBS is a multimodal deep-learning framework for transcription factor binding site (TFBS) prediction. The current repository includes:

- the training script: `cnn_multimodal_mstc_crossattn_v2_pro_2.py`
- the trained PyTorch weights: `best_model_v2_pro.pth`
- a standalone inference script: `predict.py`

## Repository contents

```text
UniAttnNet-TFBS/
├─ cnn_multimodal_mstc_crossattn_v2_pro_2.py
├─ predict.py
├─ best_model_v2_pro.pth
├─ requirements.txt
├─ .gitignore
├─ example_data/
│  ├─ demo_input.csv
│  └─ README.md
└─ outputs/
```

## Environment

Python 3.9+ is recommended.

Install dependencies:

```bash
pip install -r requirements.txt
```

## Input data format

The CSV file should contain the following columns:

- `seq_101bp`: DNA sequence (101 bp recommended)
- `Protein_seq`: protein amino-acid sequence of the TF

Optional columns:

- `TF_symbol`: used for per-TF statistics
- `label`: binary label (`0` or `1`); if present, overall and per-TF metrics will be calculated

A minimal example is provided in `example_data/demo_input.csv`.

## Standalone prediction

Use the trained model directly without retraining:

```bash
python predict.py --input_csv example_data/demo_input.csv --model_path best_model_v2_pro.pth --output_dir outputs/demo
```

Outputs:

- `predictions.csv`: per-sample predicted probability and binary prediction
- `overall_metrics.csv`: overall metrics if labels are present
- `per_tf_metrics.csv`: per-TF metrics if labels are present

## Training / external evaluation

Example command for training and evaluating on an external dataset:

```bash
python cnn_multimodal_mstc_crossattn_v2_pro_2.py --csv path/to/train.csv --external_csv path/to/external.csv --batch_size 128 --epochs 30 --lr 5e-4 --max_prot_len 800
```

## Model configuration used in the released checkpoint

- random seed: `2025`
- DNA input length: `101`
- maximum protein length: `800`
- batch size: `128`
- epochs: `30`
- learning rate: `5e-4`
- early stopping patience: `5`
- DNA kernel sizes: `(5, 9, 13)`
- protein kernel sizes: `(9, 15, 21)`
- attention heads: `4`
- optimizer: `Adam`
- weight decay: `1e-4`

## Public data note

Only a public subset / example data should be uploaded to the repository. If the full dataset cannot be released due to source restrictions, add a short note describing:

1. which subset is public,
2. where the original data came from,
3. how others can reconstruct or request the remaining data.

## Suggested manuscript statements

### Code availability

The trained model and the UniAttnNet-TFBS standalone source code are available at: `https://github.com/hanxyuuuu/UniAttnNet-TFBS`.

### Data availability

A public subset of the data and example input files are available in the GitHub repository. Additional source data supporting this study are provided with the article and/or are available from the corresponding author upon reasonable request, subject to the terms of the original data sources.

## Notes before publication

Before making the repository public, verify that it does **not** contain:

- private data
- API keys or tokens
- absolute local file paths in documentation
- unnecessary cache files / intermediate outputs

