import torch
import argparse
from nanogpt.checkpoint import load_checkpoint
from nanogpt.tokenizer import load_bpe_tokenizer, encode_text, decode_tokens


def generate_text(
    checkpoint_path,
    prompt="",
    max_new_tokens=500,
    temperature=1.0,
):
    """Generate text from a trained model checkpoint"""

    # Load checkpoint with config
    print(f"Loading checkpoint: {checkpoint_path}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, config, checkpoint_dict = load_checkpoint(checkpoint_path, device)

    # Load tokenizer
    tokenizer_path = config.data.tokenizer_path
    print(f"Loading tokenizer from {tokenizer_path}...")
    tokenizer = load_bpe_tokenizer(tokenizer_path)

    # Encode prompt
    if prompt:
        print(f"\nPrompt: {prompt}")
        token_ids = encode_text(tokenizer, prompt)
        context = torch.tensor([token_ids], dtype=torch.long, device=device)
    else:
        # Start with empty context
        context = torch.zeros((1, 1), dtype=torch.long, device=device)

    # Generate
    print(f"\nGenerating {max_new_tokens} tokens...")
    with torch.no_grad():
        generated = model.generate(context, max_new_tokens=max_new_tokens)

    # Decode
    generated_text = decode_tokens(tokenizer, generated[0].tolist())

    print("\n" + "=" * 80)
    print("GENERATED TEXT:")
    print("=" * 80)
    print(generated_text)
    print("=" * 80)

    return generated_text


def main():
    parser = argparse.ArgumentParser(description="Generate text from trained model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.pth file)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="",
        help="Text prompt to start generation (optional)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=500,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Sampling temperature"
    )

    args = parser.parse_args()

    generate_text(
        checkpoint_path=args.checkpoint,
        prompt=args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
    )


if __name__ == "__main__":
    main()
