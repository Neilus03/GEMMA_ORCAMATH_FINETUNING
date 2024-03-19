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
'''
# Check if the token is already saved
token_path = HfFolder.path_token
if not os.path.exists(token_path):
    # If the token is not saved, ask for it and save it
    access_token = input("Add your gemma HF access token: ")
    HfFolder.save_token(access_token)  # Save the token for future use
else:
    print("HF access token is already saved.")


# Assuming 'checkpoint_path' is the path to your 'checkpoint-50010' directory
checkpoint_path = "/home/ndelafuente/Desktop/GEMMA_ORCAMATH_FINETUNING/weights/Gemma-2b/e10_gemma_2b_qvko_r8_a16_lr5e-5_bs12/checkpoint-50010"

# Load the model from the Hugging Face Hub (without the adapter head)
pretrained_model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2b", # Load the model from the Hugging Face Hub (without the adapter head)
    device_map="cuda:0", # Use the default device (GPU if available, CPU otherwise)
    trust_remote_code=True, 
)

# Load the tokenizer from the local directory
tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)


# Load the adapter head from the local directory
model = PeftModel.from_pretrained(pretrained_model, checkpoint_path)

# Merge the adapter head into the model and unload it. This will make the model ready for inference. 
finetuned_model = model.merge_and_unload().to("cuda:0")

# Example of inference
input_text = "explain exponentiation to a child in demographic terms"

# Tokenize the input text and send it to the GPU
input_ids = tokenizer(input_text, return_tensors="pt").to("cuda:0")

# Generate the output of the pretrained model for the input text
outputs = pretrained_model.generate(**input_ids, max_length=400, num_return_sequences=1)
print("Pretrained model output:")
print(tokenizer.decode(outputs[0]))
print("--------------------------------------\n")
# Generate the output of the finetuned model for the input text
print("Finetuned model output:")
outputs = finetuned_model.generate(**input_ids, max_length=400, num_return_sequences=1)
print(tokenizer.decode(outputs[0]))
'''





"""
File for testing and doing inference on the model using the trained weights.
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import safetensors.torch as torchst
import torch
import os
from huggingface_hub import HfFolder

# Assuming 'checkpoint_path' is the path to your 'checkpoint-50010' directory
checkpoint_path = "/home/ndelafuente/Desktop/GEMMA_ORCAMATH_FINETUNING/weights/Gemma-2b/e10_gemma_2b_qvko_r8_a16_lr5e-5_bs12/checkpoint-50010"

# Function to load checkpoint, adjusted to work with or without PeftModel as necessary
def load_checkpoint(checkpoint_path, model):
    """
    Load the model checkpoint from a specified path, handling both .bin and SafeTensors.
    """
    bin_checkpoint = os.path.join(checkpoint_path, "pytorch_model.bin")
    safetensors_checkpoint = os.path.join(checkpoint_path, "adapter_model.safetensors")

    checkpoint_to_load = safetensors_checkpoint if os.path.exists(safetensors_checkpoint) else bin_checkpoint

    if os.path.exists(checkpoint_to_load):
        if checkpoint_to_load.endswith(".safetensors"):
            state_dict = torchst.load_file(checkpoint_to_load)
        else:
            state_dict = torch.load(checkpoint_to_load)

        model.load_state_dict(state_dict, strict=False)  # Added strict=False to handle potential mismatches
        print(f"Checkpoint loaded from {checkpoint_to_load}")
        return model
    else:
        print(f"No checkpoint found at {checkpoint_to_load}")
        return None

# Check if the HF token is already saved
token_path = HfFolder.path_token
if not os.path.exists(token_path):
    access_token = input("Add your gemma HF access token: ")
    HfFolder.save_token(access_token)
else:
    print("HF access token is already saved.")

# Load the pretrained model from Hugging Face Hub
pretrained_model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2b",
    device_map="cuda:0",
    trust_remote_code=True,
)

# Load the tokenizer from the local directory
tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

# Initialize a PeftModel or similar structure for your finetuned weights
# Note: Adjust this part according to your actual finetuned model's structure.
# If PeftModel is not correct, replace it with the appropriate class.
# For this example, we'll assume the finetuned weights are loaded into a model of the same class as the pretrained model.
finetuned_model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2b",  # Dummy initialization, will be replaced by checkpoint weights
    device_map="cuda:0",
    trust_remote_code=True,
)

# Load the checkpoint containing adapter head weights
finetuned_model = load_checkpoint(checkpoint_path, finetuned_model)

# Exit if checkpoint loading fails
if finetuned_model is None:
    print("Failed to load the finetuned model checkpoint.")
    exit()

# Ensure both models are on the same device, here assuming CUDA is available
pretrained_model.cuda()
finetuned_model.cuda()

# Example of inference
input_text = "explain exponentiation to a child in demographic terms"
input_ids = tokenizer(input_text, return_tensors="pt").to("cuda:0")

# Generate output from the pretrained model
outputs_pretrained = pretrained_model.generate(**input_ids, max_length=400, num_return_sequences=1)
print("Pretrained model output:")
print(tokenizer.decode(outputs_pretrained[0]), "\n--------------------------------------\n")

# Generate output from the finetuned model
outputs_finetuned = finetuned_model.generate(**input_ids, max_length=400, num_return_sequences=1)
print("Finetuned model output:")
print(tokenizer.decode(outputs_finetuned[0]))
