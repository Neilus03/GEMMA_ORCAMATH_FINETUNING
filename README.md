# LLM Easy Finetuning with OrcaMath

## Introduction
This project aims to simplify the finetuning process of large language models, specifically Llama-2 & Gemma-7b, using the OrcaMath dataset. The focus is on enhancing model performance for mathematical reasoning tasks.

## Requirements
- Python 3.8+
- PyTorch 1.8+
- Transformers library
- Additional dependencies listed in `requirements.txt`

## Installation
Clone this repository:
`git clone https://github.com/Neilus03/LLM_EASY_FINETUNING.git`

Install dependencies:
`cd LLM_EASY_FINETUNING`
`pip install -r requirements.txt`

## Dataset
The OrcaMath dataset is stored on Google Drive due to its size. Download it from [orcamath_data](https://drive.google.com/file/d/1JHDUPlG5igZ1QZ0McNYmUKzixV9pJXIZ/view?usp=sharing) and place it in the `data` folder.

## Usage
Run the finetuning script with:
`./run.sh`

For detailed usage and configuration options, see the `configs` directory.

## Configuration
Customize your training by modifying the configuration files in the `configs` directory. Each file corresponds to different model settings and training parameters.

## Acknowledgements
This repo is mainly based on Huawei Lin's [LLMsEasyFinetune](https://github.com/huawei-lin/LLMsEasyFinetune/tree/master?tab=readme-ov-file) repo.

