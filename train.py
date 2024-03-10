"""
This script is used to train a model using the PEFT model from the transformers library and the LoRA configuration.
"""

import os  # Provides functions to interact with the operating system, like path manipulation.
import copy  # Enables shallow and deep copy operations for complex objects.
import json  # Allows for encoding and decoding JSON data.
import logging  # Facilitates logging messages for debugging and operational purposes.
import argparse  # Supports the creation of command-line interfaces with various argument types.
import distutils.util  # Contains utility functions used by distutils modules for building Python packages.
from dataclasses import dataclass, field  # Decorators and functions for creating mutable data structures.
from typing import Dict, Optional, Sequence  # Contains type hints for static typing of collections and optional values.
from peft import (  # Includes functionality to enhance transformer models with PEFT (e.g., LoRA).
    LoraConfig,  # Configuration class for LoRA (Low-Rank Adaptation) parameters.
    get_peft_model,  # Retrieves a PEFT-enhanced transformer model.
    get_peft_model_state_dict,  # Obtains the state dictionary of a PEFT model for serialization.
    prepare_model_for_kbit_training,  # Prepares a transformer model for training with quantization.
    set_peft_model_state_dict,  # Sets the state of a PEFT model from a state dictionary.
)
from datasets import load_dataset  # Utility for loading and processing datasets in a standardized way.
import torch  # Core library for tensor computation and neural networks.
import transformers  # Hugging Face's library for NLP models and utilities.
from utils import jload  # Custom function to load JSON files.
from torch.utils.data import Dataset  # Base class for representing a dataset in PyTorch.
from transformers import Trainer, BitsAndBytesConfig  # Classes for model training and 8-bit optimization configuration.
from pathlib import Path # Object-oriented interface to file system paths.
from transformers import PreTrainedTokenizer, PreTrainedModel  # Base classes for tokenizers and models.


# Constant representing the value to ignore when calculating the loss function.
IGNORE_INDEX = -100
# Token used to pad sequences to the same length for batching purposes.
DEFAULT_PAD_TOKEN = "[PAD]"
# Token indicating the end of a sequence.
DEFAULT_EOS_TOKEN = "</s>"
# Token indicating the beginning of a sequence.
DEFAULT_BOS_TOKEN = "<s>"
# Token used to represent out-of-vocabulary (OOV) words or tokens not found in the vocabulary.
DEFAULT_UNK_TOKEN = "<unk>"
# Dictionary holding prompt templates for the dataset, facilitating context-based prompts for the model.
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

# Dataclass to store model arguments such as the model name or path.
@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")

# Dataclass to store arguments related to the data, including paths to training and evaluation datasets.
@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    eval_data_path: str = field(default=None, metadata={"help": "Path to the eval data."})

# Dataclass to store training-related arguments, extending the built-in TrainingArguments class from transformers.
@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )



def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """
    Definition:
        - Update the tokenizer and model embeddings with new special tokens. This function adds new tokens
          from special_tokens_dict to the tokenizer and updates the model'stoken embeddings to accommodate the
          new vocabulary size. It initializes new token embeddingsto the average of all previous embeddings.
    
    Args:
        - special_tokens_dict (Dict): A dictionary of special tokens to add to the tokenizer.
        - tokenizer (transformers.PreTrainedTokenizer): The tokenizer associated with the model.
        - model (transformers.PreTrainedModel): The model whose embeddings will be updated.
    
    Note: This function increases the size of the embeddings and may result in a size that is not
    divisible by 64, which is an unoptimized size for many hardware accelerators.
    """
    # Add new special tokens to the tokenizer and get the number of tokens added
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    
    # Resize model embeddings to fit the new tokenizer length
    model.resize_token_embeddings(len(tokenizer))
    
    # If new tokens were added, initialize their embeddings
    if num_new_tokens > 0:
        # Get the current input and output embeddings from the model
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data
        
        # Compute the average embedding across all existing tokens (excluding new tokens)
        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        
        # Assign the average embedding to the new token positions in the embeddings
        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """
    Definition:
        - Converts a list of strings into a format that is compatible with the transformer model. This function
          takes each string in the input list and tokenizes it using the provided tokenizer. The tokenizer
          converts each string into a sequence of token IDs, which are numerical representations of tokens.
    
    Args:
        - strings (Sequence[str]): A list of strings to tokenize.
        - tokenizer (transformers.PreTrainedTokenizer): The tokenizer that converts strings into token IDs.
    
    Returns:
        - A dictionary with the following key-value pairs:
            - 'input_ids': List of token ID sequences, each corresponding to a tokenized string.
            - 'labels': Duplicate of 'input_ids' for use as labels during model training.
            - 'input_ids_lens': List of integers indicating the length of each tokenized string sequence,
              excluding padding tokens.
            - 'labels_lens': Duplicate of 'input_ids_lens', used as lengths for the labels.
    
    Note:
        - This function assumes that the tokenizer is already configured with a maximum sequence length and
          will truncate sequences that exceed this length. It also assumes that the tokenizer is configured
          to return PyTorch tensors ('pt').
    """
    # Tokenize each string and collect the results in a list
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    
    # Extract token IDs and sequence lengths from the tokenized results
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    
    # Return a dictionary of tokenized input data
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """
    Definition:
        - Prepares and tokenizes the data for model training. The function takes pairs of source and target
          strings, concatenates each pair, and then tokenizes them. The tokenized inputs become the `input_ids`,
          and the `labels` are prepared by setting the source tokens to `IGNORE_INDEX` except for the last one,
          which is typically the end-of-sequence token.

    Args:
        - sources (Sequence[str]): A list of source strings that provide context for the task.
        - targets (Sequence[str]): A list of target strings that the model should learn to generate.
        - tokenizer (transformers.PreTrainedTokenizer): The tokenizer to convert strings to token IDs.

    Returns:
        - A dictionary with processed `input_ids` and `labels` for training the model. The `input_ids` are
          token IDs corresponding to the concatenated source and target strings, and `labels` are set up for
          language modeling where only the target part is considered for loss calculation.

    Note:
        - The function assumes that `sources` and `targets` are aligned and of the same length. It processes
          each pair in parallel, tokenizes them, and constructs `input_ids` and `labels` for supervised training.
    """

    # Concatenate each source and target string to form a full example
    examples = [s + t for s, t in zip(sources, targets)]

    # Tokenize the full examples and sources
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]

    # Extract `input_ids` from the tokenized examples
    input_ids = examples_tokenized["input_ids"]

    # Initialize `labels` with a copy of `input_ids`
    labels = copy.deepcopy(input_ids)

    # Set the tokens corresponding to the source to IGNORE_INDEX so they are not used in loss calculation
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len - 1] = IGNORE_INDEX  # Exclude the last token of the source, which is not ignored

    # Return a dictionary with `input_ids` and `labels` prepared for the model
    return dict(input_ids=input_ids, labels=labels)



class SupervisedDataset(Dataset):
    """
    A PyTorch dataset class for supervised fine-tuning on a given text dataset.
    
    This class handles the loading and preprocessing of the dataset for training language models
    in a supervised setting, where each example consists of a source text and a corresponding target text.
    The class leverages prompts to format the inputs and targets in a way that's suitable for language
    modeling tasks.
    
    Attributes:
        input_ids (torch.Tensor): The tokenized and numericalized source and target texts combined.
        labels (torch.Tensor): The labels for language modeling, where only the target part of `input_ids`
                               is used for calculating loss, with the source part being ignored.
    
    Args:
        data_path (str): The file path to the dataset. Supported formats are JSON and JSON Lines.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer used to tokenize the texts.
    """

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        # Load the dataset from the given path. Supports both JSON and JSON Lines formats.
        logging.warning("Loading data...")
        if data_path.strip().split(".")[-1] == "jsonl":
            # If the dataset is in JSON Lines format
            with open(data_path) as f:
                list_data_dict = [json.loads(line) for line in f]
        else:
            # Assume JSON format
            list_data_dict = jload(data_path)

        # Prepare the prompts for formatting the inputs
        logging.warning("Formatting inputs...")
        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        sources = [
            prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
            for example in list_data_dict
        ]
        targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]

        # Tokenize the formatted sources and targets
        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)

        # Store the tokenized input_ids and labels for training
        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        """
        Returns the number of examples in the dataset.
        """
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        """
        Retrieves the i-th example from the dataset.
        """
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])



@dataclass
class DataCollatorForSupervisedDataset(object):
    """
    A custom data collator for supervised fine-tuning of language models.
    
    This class is responsible for collating batches of data instances together into
    a single batched tensor. It pads input sequences and labels to the maximum length
    within a batch, and creates an attention mask for the inputs.
    
    Attributes:
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer used to tokenize the text,
        which also provides the `pad_token_id` for padding.
    """
    tokenizer: transformers.PreTrainedTokenizer 

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        """
        Collates a batch of instances into a single batch.
        
        This method takes a list of dictionaries, where each dictionary contains 'input_ids'
        and 'labels' for a single example, and collates them into a batch by padding the sequences
        to the same length and creating an attention mask.
        
        Args:
            instances (Sequence[Dict]): A sequence of dictionaries, each containing 'input_ids'
                and 'labels' for a single example.
        
        Returns:
            Dict[str, torch.Tensor]: A dictionary with keys 'input_ids', 'labels', and 'attention_mask'.
                - 'input_ids' and 'labels' are batched and padded tensors of the input sequences and labels.
                - 'attention_mask' is a tensor indicating which elements in 'input_ids' are padding and which are actual tokens.
        """
        # Extract input_ids and labels from the instances and batch them together
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        
        # Pad the batched input_ids and labels so that each sequence in the batch has the same length
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX  # Use a special token ID for padding in labels
        )
        
        # Create an attention_mask to indicate which tokens are padding and which are actual data
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        
        # Return the collated batch as a dictionary
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """
    Creates the data module for supervised training.

    This function prepares the dataset for both training and evaluation (if evaluation data path is provided)
    by utilizing the SupervisedDataset class. It also prepares a data collator that handles batching and data
    preprocessing steps necessary for training.

    Args:
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer used for tokenizing the input text.
        data_args: An object or structure containing the data paths for the training and optional evaluation datasets.

    Returns:
        Dict: A dictionary containing:
            - train_dataset: An instance of SupervisedDataset prepared with the training data.
            - eval_dataset (optional): An instance of SupervisedDataset prepared with the evaluation data, if provided.
            - data_collator: An instance of DataCollatorForSupervisedDataset used for preparing batches during training.
    """

    # Prepare the training dataset using the provided tokenizer and data path from data_args
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path)
    
    # Initialize the eval_dataset as None, it will be created if an evaluation data path is provided
    eval_dataset = None
    
    # If an evaluation data path is provided in data_args, prepare the evaluation dataset
    if data_args.eval_data_path is not None:
        eval_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.eval_data_path)
    
    # Create an instance of the DataCollatorForSupervisedDataset to handle batching and preprocessing
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    
    # Return a dictionary with the prepared datasets and data collator for use in training
    return dict(
        train_dataset=train_dataset,  # The training dataset
        eval_dataset=eval_dataset,  # The evaluation dataset, if provided
        data_collator=data_collator  # The data collator for handling batch preparation
    )


def load_checkpoint(
    checkpoint_path: str,
    model: PreTrainedModel
) -> Optional[PreTrainedModel]:
    """
    Load the model checkpoint from a specified path. This function handles both traditional
    PyTorch .bin files and Huggingface SafeTensors.

    Args:
        checkpoint_path (str): Path to the directory containing checkpoint files.
        model (PreTrainedModel): The model to which the checkpoint should be loaded.
        use_safetensors (bool): Flag indicating whether to load from SafeTensors. If False, will load from .bin.

    Returns:
        Optional[PreTrainedModel]: The model with loaded weights if successful, None otherwise.
    """
    # Construct the filenames for bin and safetensors files.
    bin_checkpoint = os.path.join(checkpoint_path, "pytorch_model.bin")
    safetensors_checkpoint = os.path.join(checkpoint_path, "adapter_model.safetensors")

    # Determine which file to load based on the use_safetensors flag and file existence.
    checkpoint_to_load = safetensors_checkpoint if os.path.exists(safetensors_checkpoint) else bin_checkpoint
    
    if os.path.exists(checkpoint_to_load):
        # Load the weights from the file into a state dictionary.
        state_dict = torch.load(checkpoint_to_load)
        
        # Set the model's state dictionary to the loaded state dictionary, using set_peft_model_state_dict to handle PEFT models.
        set_peft_model_state_dict(model, state_dict)
        
        print(f"Checkpoint loaded from {checkpoint_to_load}")
        return model
    else:
        #If no checkpoint is found, return None and print a message
        print(f"No checkpoint found at {checkpoint_to_load}")
        print("Returning the original model and training from scratch.")
        return None


def train():
    # Parse command-line arguments related to model, data, and training configurations.
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args, other_args = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    
    # Setup additional parser for custom arguments not covered by Hugging Face's ArgumentParser.
    other_parser = argparse.ArgumentParser()
    
    # Define additional arguments for model quantization and LoRA (Low-Rank Adaptation).
    other_parser.add_argument('--load_in_4bit', type=lambda x:bool(distutils.util.strtobool(x)), default=False)
    other_parser.add_argument('--load_in_8bit', type=lambda x:bool(distutils.util.strtobool(x)), default=False)
    other_parser.add_argument('--use_lora', type=lambda x:bool(distutils.util.strtobool(x)), default=False)
    other_parser.add_argument('--lora_r', type=int, default=8)
    other_parser.add_argument('--lora_alpha', type=int, default=16)
    other_parser.add_argument('--lora_target_modules', nargs="+", default=['q_proj, v_proj'])
    other_args = other_parser.parse_args(other_args)
    print(other_args, other_args.use_lora)
    
    # Optional model quantization setup if requested via command line.
    bnb_config = None
    if other_args.load_in_8bit == True or other_args.load_in_4bit == True:
        load_in_4bit = other_args.load_in_4bit
        load_in_8bit = False if load_in_4bit else other_args.load_in_8bit
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

    print(f"model: {model_args.model_name_or_path}")
    
    # Load the model and tokenizer with the specified configurations.
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        quantization_config=bnb_config,
    )
    if other_args.load_in_8bit == True or other_args.load_in_4bit == True:
        model = prepare_model_for_kbit_training(model)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    # Add any missing special tokens to the tokenizer and resize model embeddings to accommodate them.
    special_tokens_dict = {token: value for token, value in [
        ("pad_token", DEFAULT_PAD_TOKEN),
        ("eos_token", DEFAULT_EOS_TOKEN),
        ("bos_token", DEFAULT_BOS_TOKEN),
        ("unk_token", DEFAULT_UNK_TOKEN)
    ] if getattr(tokenizer, token) is None}

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )
    
    # Optionally apply LoRA adaptation to the model.
    if other_args.use_lora == True:
        config = LoraConfig(
            r=other_args.lora_r,
            lora_alpha=other_args.lora_alpha,
            target_modules = other_args.lora_target_modules,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        print("Use LoRA:", config)
    
        model.enable_input_require_grads()
        model = get_peft_model(model, config)
        model.print_trainable_parameters()

        model.config.use_cache = False
        model.is_parallelizable = True
        model.model_parallel = True

    # Attempt to resume training from a checkpoint if specified.
    if training_args.resume_from_checkpoint:
        # Call the function with the appropriate arguments
        checkpoint_dir = training_args.resume_from_checkpoint
        loaded_model = load_checkpoint(checkpoint_dir, model)
        
        # Check if the model was loaded successfully
        if loaded_model is not None:
            model = loaded_model
        else:
            print("Failed to load the checkpoint.")

    # Prepare datasets and the data collator.
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)

    # Compile the model with torch.compile for efficiency if supported and LoRA is used.
    if other_args.use_lora and torch.__version__ >= "2":
        model = torch.compile(model)

    # Start the training process.
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    #Execute the training process.
    train()
