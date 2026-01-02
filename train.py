from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

# ------- CONFIGURATION -------

max_seq_length = 2048       # Tells how many words can the model read at once
dtype = None                # Auto-detect RTX 5060
load_in_4bit = True         # Force 4bit to save VRAM

# ------- LOAD MODEL -------

print(f"Loading Llama model...")

model, tokenizer = FastLanguageModel.from_pretrained(               # model: Download the base Llama model , tokenizer: Converts words to numbers (ex: Hello -> [101] [202])
                                                                    
    model_name = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = dtype,                            
    load_in_4bit = load_in_4bit
)

# ------- ADD LORA ADAPTERS ("Learning layers") -------

model = FastLanguageModel.get_peft_model(
    model, 
    r = 16,                                                                                                         # Control how much new information the model can learn (Higher = More VRAM, Longer training, Better results)    
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",],                # Specify which layers to adapt
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
)

# ------- PREPARE CUSTOM DATA -------
# This template matches Llama 3 format roughly, adapted for Alpaca style data inputs

alpaca_prompt = """Below is an instruction that describes a task. paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

# !!! IMPORTANT: This loads the "knowledge.json" file from the same folder !!! 

dataset = load_dataset("json", data_files = "knowledge.json", split = "train")

def formatting_prompts_func(examples):

    instructions =  examples["instruction"]
    inputs =        examples["input"]
    outputs =       examples["output"]
    texts = []

    for instruction, input, output in zip(instructions, inputs, outputs):

        # Must add EOS_TOKEN so the model knows when to stop generating
        text = alpaca_prompt.format(instruction, input, output + tokenizer.eos_token)   # Format the prompt using the template
        texts.append(text) 

    return {"text" : texts,}
    
dataset = dataset.map(formatting_prompts_func, batched = True)

# ------- TRAINING SETUP (The teacher) -------

print("Start training...")

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,                                  # Tokenizer to convert text to numbers
    train_dataset = dataset,                                # Training dataset
    dataset_text_field = "text",                            # Field in the dataset containing the text
    max_seq_length = max_seq_length,                        # Max sequence length for training
    dataset_num_proc = 2,                                   # Number of processes for data loading
    formatting_func = formatting_prompts_func,              # Function to format prompts
    packing = False,                                        # Disable packing for simplicity
    args = TrainingArguments( 
        per_device_train_batch_size = 2,                    # Lower for less VRAM usage
        gradient_accumulation_steps = 4,                    # Simulate larger batch size
        warmup_steps = 5,                                   # Warmup steps for learning rate
        max_steps = 60,                                     # For testing purposes, set low. Increase for real training
        learning_rate = 2e-4,                               # Learning rate
        fp16 = not torch.cuda.is_bf16_supported(),          # Use FP16 if BF16 not supported
        bf16 = torch.cuda.is_bf16_supported(),              # Use BF16 if supported
        logging_steps = 1,                                  # Log every step
        optim = "adamw_8bit",                               # Use 8-bit Adam optimizer
        weight_decay= 0.01,                                 # Weight decay for regularization
        lr_scheduler_type = "linear",                       # Linear learning rate scheduler
        seed = 3407,                                        # Random seed for reproducibility
        output_dir = "outputs",                             # Output directory
    ),
)

# ------- START TRAINING -------
trainer_stats = trainer.train()

# ------- SAVE THE MODEL -------
# This creates the "lora_model" folder that you will use in inference.py

print("Saving the LoRA adapters to 'lora_model' folder...")
model.save_pretrained("lora_model")
tokenizer.save_pretrained("lora_model")

print("Training complete.")