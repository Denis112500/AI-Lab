import torch
from unsloth import FastLanguageModel
from transformers import TextStreamer

# Setup the Model Name 
model_name = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"  # !!! Helping the GPU to actually download the model !!!
max_seq_length = 2048
dtype = None           # Half Precision method (Uses only half of the VRAM it's supposed to use) In this case 2.5 out of 5
load_in_4bit = True    # Load the model in 4bit precision (Even lower VRAM usage, but slower)



# 1.Standard pipeline: Load Brain -> Prepare Input -> Think -> Speak
print(f"Loading {model_name}...")

model, tokenizer = FastLanguageModel.from_pretrained(   # Library 
    model_name = model_name,
    max_seq_length = max_seq_length,
    dtype = dtype,                            #Auto-detect (Float16 for RTX 5060)
    load_in_4bit = load_in_4bit
)
print("Model loaded succesfully")

FastLanguageModel.for_inference(model)   # Put the model in inference mode (No gradients, no training, just answering)

print(f"Model loaded on {torch.cuda.get_device_name(0)}") # Print the GPU name
print(f"VRAM used: {torch.cuda.memory_allocated()/1024**3:.2f} GB")  # Print the VRAM used

print("-" * 50)
print("Chatbot is ready! Type 'exit' to stop.")
print("-" * 50)

text_streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)  # Streamer to print the output in real-time
system_prompt = "You are a helpful AI assistnat running locally on a RTX 5060."      # System psystem_prompt = "You are a helpful AI assistnat running locally on a RTX 5060."      # System prompt to guide the model's behavior

stop_token_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")  # Define stop token to end generation


while True:
    user_input = input("\nUser: ")

    if user_input.lower() == "exit":
        print("Exiting chat...")
        break
    
    #2.Preparing Input (Translation)
    messages = [
        {"role": "system", "content": system_prompt},  # System prompt to guide the model's behavior
        {"role": "user" , "content": user_input}   # Organized in a structured list (USER vs AI)
    ]

    inputs = tokenizer.apply_chat_template(
        conversation = messages,                        # The structured list
        tokenize = True,                                # Tokenizes the input text
        add_generation_prompt = True,                   # Adds extra tokens to signal the model to generate a response
        return_tensors = "pt"                           # pt => PyTorch ( It takes the tagged strings made by the tokenizer and puts them in a block to create Tensors )
    ).to("cuda")                                        # Move tensors to GPU

    #Generating Response
    print("Generating response...")

    _ = model.generate(
        input_ids = inputs,                     # "model_inputs" is the container of the tensor and "input_ids" are the integers inside the tensor [<-Tensor [101]<-Integer ]
        streamer = text_streamer,               # Streamer to print the output in real-time
        max_new_tokens = 2048,                   # Max of words that the model can generate
        pad_token_id=tokenizer.eos_token_id,    # Padding token to avoid errors
        temperature = 0.3,                      # Controls how creative the model can be | To low => robotic/factual responses | To High => Total gibberish   //Tested\\
        top_p = 0.9,                            # Nucleus sampling to enhance response quality
        repetition_penalty = 1.25,              # Penalizes repetition to enhance response diversity
        do_sample = True,                       # Enables sampling for more varied responses
        use_cache = True                        # Speeds up generation by caching past key values
    )

    #Decoding Response (Translation back to human language)
    
    print("-" * 30)