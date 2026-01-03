# Llama 3 Fine-Tuning with Unsloth

## Core Concepts (The "Why")

**Hardware:**  This project initially started on an **AMD GPU (RX 6600)** using the ROCm platform. However, due to driver issues and knowing that Nvidia is better in this kind of work, the final training workflow was migrated to an **Nvidia RTX 5060** to leverage the native CUDA support in Unsloth.

*  Note: The "Troubleshooting" section at the bottom of this README preserves the fixes discovered during the AMD phase for anyone attempting this on Radeon hardware.

* This project is made for self-development and learning purposes.

* This repository documents my experiments fine-tuning **Llama-3 8B** on consumer hardware (originally tested on AMD, finalized on Nvidia RTX 5060).

* The goal was to create a local AI assistant that "knows who it is" using **LoRA (Low-Rank Adaptation)** and the **Unsloth** library for optimization.

* Before running the code, it is important to understand the specific parts of the "brain" we are training. This script uses **LoRA (Low-Rank Adaptation)**, which freezes the main model and only trains small "adapter" layers. This makes training much faster and requires less memory.

## Project Structure

* **`train.py`**: The "Teacher" script. It loads the base model, applies LoRA adapters, and fine-tunes it on custom data.
* **`inference.py`**: The "Chatbot" script. It loads the saved adapters and lets you chat with the model in real-time.
* **`knowledge.json`**: The dataset. A custom JSON file containing the identity and knowledge I wanted the AI to learn.

# Theory 

## The Attention Mechanism (Target Modules)
In the code, you will see `target_modules` like `q_proj`, `k_proj`, etc. These represent the **Attention Mechanism**, which allows the model to process context. Imagine the model is a librarian:

* **`q_proj` (Query):** The **Question** the model asks.
    * *Analogy:* The librarian asks, "I need a book about..."
    
* **`k_proj` (Key):** The **Label** on every piece of information in the database.
    * *Analogy:* The spine label on a book that says "History" or "Cooking"
    .
* **`v_proj` (Value):** The **Content** of the information.
    * *Analogy:* The actual text inside the book that the librarian reads after finding a match.

* **`o_proj` (Output):** The **Result**.
    * *Analogy:* The librarian handing you the information you requested.

### Other Modules
* **`gate_proj`, `up_proj`, `down_proj`:** These are the **Feed Forward Layers** (the "thinking" layers). After the attention mechanism gathers information, these layers process it to reach a conclusion.

# Configuration & Model Loading

* This section imports necessary libraries and sets up the base model.

## Key Configurations
* **`max_seq_length = 2048`**: This is the model's **Context Window**. It determines how much text (prompt + response) the model can "remember" at one time.
    * *Note:* Higher numbers allow for longer conversations but require more VRAM.

* **`load_in_4bit = True`**: This is **Quantization**.
    * Standard models use 16-bit or 32-bit precision for numbers.
    
    * We force **4-bit** precision, which shrinks the model size significantly (like compressing a massive image) so it fits on consumer GPUs (like an RTX 3060/4060) without losing much intelligence.

## Loading the Model
```python
# model, tokenizer = FastLanguageModel.from_pretrained(...)
```
## LoRA Adapter Configuration

* This step is where the "learning" magic happens. Instead of retraining the entire brain (which is huge), we attach small, trainable adapter layers to specific parts of the model.

```python
# model = FastLanguageModel.get_peft_model(...)
```

## Advanced LoRA Parameters
* **`r = 16 (Rank):`** 
    * This determines the "size" of the new information the model can learn.

* **`Analogy`**: 
    * Think of this as the size of the notebook the librarian uses to take notes.

`16 is a standard balance. Higher numbers (e.g., 64, 128) allow learning more complex details but use much more VRAM and take longer to train.`

* **`target_modules = [...]`**: 
    * This list tells the code exactly where to attach the adapters.

    * We attach them to the Attention Mechanism **(q_proj, k_proj, v_proj, o_proj)** and the Feed Forward Layers **(gate_proj, up_proj, down_proj)** to ensure the model learns new behaviors thoroughly.

* **`lora_alpha = 16`**: 
    * This is a scaling factor for how much weight the new adapters have compared to the old model.

    * Usually, we keep alpha equal to r (1:1 ratio) or 2 * r.

* **`use_gradient_checkpointing = "unsloth"`**: 
    * A memory-saving trick.

    * It throws away some intermediate calculations during the "forward pass" and   re-calculates them during the "backward pass."

    * Result: You can fit larger models into less VRAM, but training is slightly slower (approx. 20%).

* **`lora_dropout = 0`:** 
    * This is a technique to prevent the model from memorizing the data too closely ("overfitting").

    * It randomly "turns off" some learning connections during training. Setting it to **0** means we are using all connections all the time (faster, but higher risk of overfitting on small data).

* **`bias = "none"`:** 
    * This specifies if the bias parameters (extra "nudge" values in the math) should be trained.

    * **"none"** is standard for LoRA to keep the model small and fast.

* **`random_state = 3407`:** 
    * The "Seed" for randomness.

    * Computers aren't truly random. By setting a specific number (3407 is a popular lucky number in the AI community), we ensure that if you run this code twice, you get the exact same result.

* **`use_rslora = False`:** 
    * Rank-Stabilized LoRA.

    * A different mathematical way to scale the adapters. We are using the standard LoRA method here, so this is **False**.
    
* **`loftq_config = None`:** 
    * LoftQ (LoRA + Quantization).

    * A special initialization method that can help when training very deep 4-bit models. We are using standard 4-bit loading, so we don't need this specific config.

## Data Preparation & Formatting

* This section loads your custom data (`knowledge.json`) and reshapes it into a format the model can understand.

### The Alpaca Prompt Template
* We use a specific text template called "Alpaca." It structures the data so the model knows clearly what is an instruction and what is the expected response.

```python
alpaca_prompt = """Below is an instruction...
### Instruction: {}
### Input: {}
### Response: {}"""

## {} placeholders: These are where your actual data (Instruction, Input, Output) will be inserted.

## dataset = load_dataset("json", data_files = "knowledge.json", split = "train")

    ## load_dataset: This pulls your raw JSON data into memory.
```

* **`def formatting_prompts_func(examples):`**
    ...
    * **`text = alpaca_prompt.format(instruction, input, output + tokenizer.eos_token)`**

    * formatting_prompts_func: This function runs through every row of your data and fills in the Alpaca template.

    * tokenizer.eos_token (CRITICAL):

        * This stands for End of Sentence Token.

        * We append this to the end of the output. Without it, the model would not know when to stop generating text and would ramble on forever during inference.

* **`for instruction, input, output in zip(instructions, inputs, outputs):`**
    * The zip function grabs the first item from each list and bundles them together, then the second item from each list, and so on.


## General Trainer Settings

* **`dataset_num_proc = 2`:** 
    * This uses 2 CPU cores to process the data faster before feeding it to the GPU.

* **`packing = False`:**

    * True: Combines multiple short examples into one long sequence to train faster.

    * False: Keeps examples separate. We use False here for simplicity and to prevent context bleeding between different unrelated examples.

## Training Arguments (Hyperparameters)

* Inside TrainingArguments, we define the specific rules for the training run:

* **`per_device_train_batch_size = 2`:** 
    * The number of conversation examples the GPU looks at strictly at one time.

    * Why: Kept low (2) to minimize VRAM usage.

* **`gradient_accumulation_steps = 4`:** 
    * A trick to simulate a larger batch size.

    * Math: 2 (batch size) * 4 (accumulation) = 8 effective examples processed before the model updates its weights. This stabilizes training.

* **`warmup_steps = 5`:** 
    * The model starts with a very low learning rate and slowly ramps up over 5 steps.

    * Why: This prevents the model from getting "shocked" by new data at the very beginning, which can ruin the training stability.

* **`max_steps = 60`:** 
    * The total number of training steps to run.

    * Note: 60 is a very short run for testing. For a real model, you would typically train for hundreds of steps or usage num_train_epochs = 1.

* **`learning_rate = 2e-4`:** 
    * The "speed" of learning.

    * Context: 2e-4 (0.0002) is the standard "sweet spot" for LoRA.

* **`fp16 / bf16`:** 
    * Floating Point Precision.

    * The code automatically detects if your GPU supports BF16 (Brain Floating Point, better stability). If not, it falls back to FP16.

* **`logging_steps = 1`:**
    * The trainer will print the loss (error rate) after every single step so you can monitor progress closely.

* **`optim = "adamw_8bit"`:** 
    * The optimizer algorithm.

    * We use the 8-bit version of AdamW. This reduces the memory needed for optimizer states by ~75%, allowing you to fine-tune larger models on smaller cards.

* **`weight_decay = 0.01`:**    
    * A regularization technique.

    * It adds a small penalty for having very large weights, which helps prevent the model from overfitting.

* **`lr_scheduler_type = "linear"`:**

    * After the warmup, the learning rate will linearly decrease (fade out) until it hits 0 at the end of training. This helps the model "settle" into the best solution.

* **`seed = 3407`:** 
    * The random seed ensures that the data shuffling and weight initialization are identical every time you run the script.

* **`output_dir = "outputs"`:** 
    * The folder where checkpoints (saves during training) will be stored.

## Saving the Model

* **`After training is complete, the script saves the new "LoRA Adapters" to your disk.`**

    * model.save_pretrained("lora_model")
    * tokenizer.save_pretrained("lora_model")

    * lora_model folder: This will contain the small adapter files (usually < 100MB).

* Usage: You will point your inference script (e.g., inference.py) to this folder to load your fine-tuned chatbot.

# Troubleshooting & Errors
    
* **`Security Error (apt vs. dpkg)`**

    * **The Issue:** When installing local .deb files via sudo apt install ./file.deb, Ubuntu switches to the _apt user. This user is sandboxed and cannot access files inside personal /home/ directories.

    * **The Fix:** Use dpkg (which runs as root and ignores the sandbox) followed by a dependency fix.
        Bash

        * `Force install the deb file`
            sudo dpkg -i amdgpu-install_6.2.60200-1_all.deb

        * `Fix missing dependencies immediately after`
            sudo apt install -f -y

* **`RuntimeError: HIP Error (Navi 21 vs Navi 23)`**

    * **The Issue:** The AMD RX 6600 (Navi 23) is not officially supported by older ROCm versions, which default to Navi 21 (RX 6800/6900). Attempting to run AI workloads results in a crash.

    * **The Fix:** Override the graphics version to "trick" the software into recognizing the card as a supported architecture (RDNA 2).

        * `Temporary (One session , had to run this everytime i started the project)`
            export HSA_OVERRIDE_GFX_VERSION=10.3.0


# Related tools

* I built a custom PHP/HTML Knowledge Base to log these specific "Runtime" and "Security" errors.
* It helps track the exact terminal commands and workarounds required to reproduce this environment in the future.

# Acknowledgements
* **Unsloth Library:** For making fine-tuning faster.
* **Generative AI:** Used for code assistance, drafting documentation, and summarizing theoretical concepts