import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig
from trl import SFTTrainer

# Setup the Model Name 
model_name = "openbmb/MiniCPM-2B-sft-bf16"  # !!! Helping the GPU to actually download the model !!!

# Loading the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name) # !!!! The tokinezer is the tool that turns text into numbers so the machine can understand !!!!

 # Filling the gaps in so the code doesn't crash. 
tokenizer.pad_token = tokenizer.eos_token

#   bnb_config = BitsAndBytesConfig(
#   load_in_4bit = True,
#   bnb_4bit_quant_type = "nf4",
#   bnb_4bit_compute_dtype = torch.float16,
#   )

print(f"Loading {model_name}...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map = "auto",
    trust_remote_code = True
)
print("Model loaded succesfully")