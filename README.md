# Gemma Finetuning with OrcaMath dataset
<p align="left">
  <img src="https://github.com/Neilus03/GEMMA_ORCAMATH_FINETUNING/assets/87651732/0beadfe5-b996-4cc4-8f5d-fb17756ebd8b" alt="First Image" width="400" height="220"/>
  <img src="https://github.com/Neilus03/GEMMA_ORCAMATH_FINETUNING/assets/87651732/727eca80-b98d-4780-a4f8-b953e80d7ff4" alt="Second Image" width="400"/>
</p>

## Introduction
This project aims to simplify the finetuning process of large language models, specifically Gemma-7b and Gemma-2b, using the OrcaMath dataset. The focus is on enhancing model performance for mathematical reasoning tasks. 
 

## Requirements 
- Python 3.8+
- PyTorch 1.8+
- Transformers library 
- Additional dependencies listed in `requirements.txt`

## Installation
Clone this repository:

`git clone https://github.com/Neilus03/GEMMA_ORCAMATH_FINETUNING.git`
 
Install dependencies:
```bash
cd GEMMA_ORCAMATH_FINETUNING 
```
```bash
pip install -r requirements.txt
```

## Dataset
The OrcaMath dataset as a json file ready to be trained is stored on Google Drive due to its size. Download it from [orcamath_data](https://drive.google.com/file/d/1JHDUPlG5igZ1QZ0McNYmUKzixV9pJXIZ/view?usp=sharing) and place it in the `data` folder, substituting the `placeholder.json` file

RAW data in parquet format can be found [here](https://huggingface.co/datasets/microsoft/orca-math-word-problems-200k/tree/main/data)

## Usage

### Finetuning
Run the finetuning script with:
`./run.sh`

For detailed usage and configuration options, see the `configs` directory and `run.sh` file.

### Inference
Modify prompt in the `inference.py` file's `input_text` variable and run `python inference.py` on the command line.

## Configuration
Customize your training by modifying the configuration files in the `configs` directory. Each file corresponds to different model settings and training parameters.

## Acknowledgements
This repo is mainly inspired on Huawei Lin's [LLMsEasyFinetune](https://github.com/huawei-lin/LLMsEasyFinetune/tree/master?tab=readme-ov-file) which did a similar process for different hardware, dataset and model.

