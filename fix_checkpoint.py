import torch

# Load checkpoint
checkpoint = torch.load("emergency_checkpoint_step_1465.pt")

# Remove "module." prefix from DDP
state_dict = checkpoint["model_state_dict"]
new_state_dict = {}
for key, value in state_dict.items():
    if key.startswith("module."):
        new_key = key[7:]  # Remove "module." prefix
        new_state_dict[new_key] = value
    else:
        new_state_dict[key] = value

checkpoint["model_state_dict"] = new_state_dict

# Save fixed checkpoint
torch.save(checkpoint, "emergency_checkpoint_step_1465_fixed.pt")
print("Fixed checkpoint saved!")
