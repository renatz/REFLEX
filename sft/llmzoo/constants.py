# constants.py
IGNORE_INDEX = -100

# Token definitions for different model families
TOKENS = {
    "llama_mistral": {
        "DEFAULT_PAD_TOKEN": "<pad>",
        "DEFAULT_BOS_TOKEN": "<s>",
        "DEFAULT_EOS_TOKEN": "</s>",
        "DEFAULT_UNK_TOKEN": "<unk>",
    },
    "qwen": {
        "DEFAULT_PAD_TOKEN": "<|endoftext|>",
        "DEFAULT_BOS_TOKEN": "<|im_start|>",
        "DEFAULT_EOS_TOKEN": "<|im_end|>",
        "DEFAULT_UNK_TOKEN": "null",  # Qwen doesn't use unk token
    }
}

# Default exports for backward compatibility (can be removed if not needed)
DEFAULT_PAD_TOKEN = TOKENS["llama_mistral"]["DEFAULT_PAD_TOKEN"]
DEFAULT_BOS_TOKEN = TOKENS["llama_mistral"]["DEFAULT_BOS_TOKEN"]
DEFAULT_EOS_TOKEN = TOKENS["llama_mistral"]["DEFAULT_EOS_TOKEN"]
DEFAULT_UNK_TOKEN = TOKENS["llama_mistral"]["DEFAULT_UNK_TOKEN"]