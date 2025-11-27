import torch
import sys

# Just load and test the checkpoint
print("Loading checkpoint...")
checkpoint = torch.load("emergency_checkpoint_step_1465.pt", map_location="cpu")

print(f"Step: {checkpoint['step']}")
print(f"Loss: {checkpoint['loss']:.4f}")

# Check keys
state_dict = checkpoint["model_state_dict"]
sample_keys = list(state_dict.keys())[:5]
print(f"\nSample keys from checkpoint:")
for key in sample_keys:
    print(f"  {key}")

# Check if DDP wrapped
has_module_prefix = any(k.startswith("module.") for k in state_dict.keys())
print(f"\nHas module. prefix (DDP): {has_module_prefix}")

# Import the GPT model from train_gpt2
sys.path.insert(0, "/home/ubuntu/code/neuralnets")
from train_gpt2 import GPT, GPTConfig

# Create model
config = GPTConfig(vocab_size=50304)
model = GPT(config)

# Fix state dict if needed
if has_module_prefix:
    print("Removing module. prefix...")
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("module.", "", 1)
        new_state_dict[new_key] = value
    state_dict = new_state_dict

# Load weights
print("Loading weights into model...")
model.load_state_dict(state_dict)
print("SUCCESS! Model loaded.")

# Quick inference test
import tiktoken
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

enc = tiktoken.get_encoding("gpt2")
prompt = "Once upon a time"
tokens = enc.encode(prompt)
x = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)

print(f"\nGenerating from prompt: '{prompt}'")
print("-" * 50)

with torch.no_grad():
    for _ in range(50):
        logits, _ = model(x)
        logits = logits[:, -1, :]
        probs = torch.nn.functional.softmax(logits / 0.8, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        x = torch.cat([x, next_token], dim=1)

generated_text = enc.decode(x[0].tolist())
print(generated_text)
