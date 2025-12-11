import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from peft import LoraConfig
from trl import SFTTrainer

# Setup the Model Name 
model_name = "openbmb/MiniCPM-2B-sft-bf16"  # !!! Helping the GPU to actually download the model !!!

# Loading the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name) # !!!! The tokinezer is the tool that turns text into numbers so the machine can understand !!!!

# Filling the gaps in so the code doesn't crash.                [101, 202]                                  [101, 202]
#                                                               [303]       <- Jagged edge (GPU crash)      [303, 000] <- Filled with padding
tokenizer.pad_token = tokenizer.eos_token

#   bnb_config = BitsAndBytesConfig(
#   load_in_4bit = True,
#   bnb_4bit_quant_type = "nf4",
#   bnb_4bit_compute_dtype = torch.float16,
#   )

# 1.Standard pipeline: Load Brain -> Prepare Input -> Think -> Speak
print(f"Loading {model_name}...")
model = AutoModelForCausalLM.from_pretrained(   # Library 
    model_name,
    dtype=torch.float16,        # Half Precision method (Uses only half of the VRAM it's supposed to use) In this case 2.5 out of 5
    device_map = "auto",        # Find the GPU automatically
    trust_remote_code = True    # The model I use has a custom code and this line let's me run it's special setup files
)
print("Model loaded succesfully") 

# 2.Preparing Input (Translation)
messages = [
    {"role": "user" , "content": "Make a poem about Linda"}   # Organized in a structured list (USER vs AI)
]

#   APPLY_CHAT_TEMPLATE : Converts the list we gave above into a string that the model understands
#   It adds special flags like <|user|> or <|assistant|>
chat_text = tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt = True)

#   tokenizer(chat_text, return_tensors = "pt").to(model.device) "Translator" -> Turns the tagged string into a list of numbers that the GPU can process (Tensors)/(The Rectangles)
#   [101, 202]
#   [303, 000] -> This is a Tensor
model_inputs = tokenizer(chat_text, return_tensors = "pt").to(model.device)     # pt => PyTorch (It takes the tagged strings made by the tokenizer and puts them in a block to create Tensors)

print("Generating response... (This might take a few seconds)")

generated_ids = model.generate(
    model_inputs.input_ids,     # "model_inputs" is the container of the tensor and "input_ids" are the integers inside the tensor [<-Tensor [101]<-Integer ]
    max_new_tokens = 200,       # Max of words that the model can generate
    do_sample = True,           # Allows the model to be creative
    temperature = 0.7,          # Controls how creative the model can be | To low => robotic/factual responses | To High => Total gibberish   //Tested\\
    use_cache = False           # Specific error fix for this model (MiniCPM)
)

response = [
    tokenizer.decode(output_ids, skip_special_tokens = True)    #The model outputs TokenIDs (numbers) and translates them back into human language / skip_special_tokens = True , Hides internal flags (</s> <|> etc..)
    for output_ids in generated_ids         # The loop takes out a piece [output_ids(numbers)] from the container (generated_ids) and sends them to the decoder
]

print("-" * 30)
print(response[0])      # Printing final response
print("-" * 30)