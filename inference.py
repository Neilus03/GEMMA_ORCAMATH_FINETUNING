"""
File for testing and doing inference on the model using the trained weights.

"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import safetensors.torch as torchst
import torch
from peft import PeftModel
import torch
import os
from huggingface_hub import HfFolder

# Check if the token is already saved
token_path = HfFolder.path_token
if not os.path.exists(token_path):
    # If the token is not saved, ask for it and save it
    access_token = input("Add your gemma HF access token: ")
    HfFolder.save_token(access_token)  # Save the token for future use
else:
    print("HF access token is already saved.")


# Assuming 'checkpoint_path' is the path to your 'checkpoint-50010' directory
checkpoint_path = "./weights/Gemma-2b/e10_gemma_2b_qvko_r8_a16_lr5e-5_bs12/checkpoint-50010"

# Load the model from the Hugging Face Hub (without the adapter head)
pretrained_model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2b", # Load the model from the Hugging Face Hub (without the adapter head)
    device_map="cuda:1", # Use the default device (GPU if available, CPU otherwise)
    trust_remote_code=True, 
)

# Load the tokenizer from the local directory
tokenizer_finetuned = AutoTokenizer.from_pretrained(checkpoint_path)

# Load the adapter head from the local directory
finetuned_model = PeftModel.from_pretrained(pretrained_model, checkpoint_path)

# Merge the adapter head into the model and unload it. This will make the model ready for inference. 
finetuned_model = finetuned_model.merge_and_unload().to("cuda:1")

# Example of inference
input_text = "How many pairs of sides are parallel in a regular hexagon?"

# Tokenize the input text and send it to the GPU

input_ids_for_finetuned = tokenizer_finetuned(input_text, return_tensors="pt").to("cuda:1")

# Generate the output of the finetuned model for the input text
print("Finetuned model output:")
outputs = finetuned_model.generate(**input_ids_for_finetuned, max_length=1000, num_return_sequences=1)
print(tokenizer_finetuned.decode(outputs[0]))
