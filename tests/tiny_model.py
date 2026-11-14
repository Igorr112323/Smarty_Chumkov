from pathlib import Path

VOCAB = [
    "[PAD]",
    "[UNK]",
    "[CLS]",
    "[SEP]",
    "[MASK]",
] + [chr(code) for code in range(0x430, 0x450)] + [
    "завод",
    "год",
    "году",
    "москва",
    "деталь",
    "детали",
    "выпустил",
    "основана",
    "город",
    "который",
    "князь",
    "в",
    "и",
    "не",
    "было",
    "выпускал",
]


def build_tiny(directory, seed=0):
    import torch
    from transformers import BertConfig, BertModel, BertTokenizerFast

    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    vocab_file = target / "vocab.txt"
    vocab_file.write_text("\n".join(VOCAB) + "\n", encoding="utf-8")
    tokenizer = BertTokenizerFast(vocab_file=str(vocab_file), do_lower_case=True)
    tokenizer.save_pretrained(str(target))
    torch.manual_seed(seed)
    config = BertConfig(
        vocab_size=len(VOCAB),
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=512,
        attn_implementation="eager",
    )
    model = BertModel(config).eval()
    model.save_pretrained(str(target))
    return target
