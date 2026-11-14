import hashlib
import json
from pathlib import Path

LOCAL_FILE = "local_encoder.pt"
MODEL_NAME = "spanver-char-encoder"
THREADS = 4
SEED = 0
SPECIALS = ("[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]")
CHARS = (
    "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789"
    " .,!?;:()[]{}<>\"'«»-–—/%№=+*"
)
DEFAULT_CONFIG = {"dim": 128, "layers": 3, "heads": 4, "ff": 256, "positions": 512}


def runtime():
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("torch не установлен") from error
    torch.set_num_threads(THREADS)
    torch.manual_seed(SEED)
    return torch


def build_vocab():
    tokens = list(SPECIALS)
    for char in CHARS:
        if char not in tokens:
            tokens.append(char)
    return tokens


class Tokenizer:
    def __init__(self, vocab):
        self.vocab = list(vocab)
        self.index = {token: position for position, token in enumerate(self.vocab)}
        self.pad_id = self.index["[PAD]"]
        self.unk_id = self.index["[UNK]"]
        self.cls_id = self.index["[CLS]"]
        self.sep_id = self.index["[SEP]"]
        self.mask_id = self.index["[MASK]"]

    def ids(self, text, limit):
        found = []
        for char in (text or "").lower():
            found.append(self.index.get(char, self.unk_id))
            if len(found) >= limit:
                break
        return found


def build_encoder(torch, config):
    nn = torch.nn

    class Block(nn.Module):
        def __init__(self, dim, heads, ff):
            super().__init__()
            self.attn = nn.MultiheadAttention(dim, heads, batch_first=True, dropout=0.0)
            self.norm1 = nn.LayerNorm(dim)
            self.norm2 = nn.LayerNorm(dim)
            self.ff = nn.Sequential(nn.Linear(dim, ff), nn.GELU(), nn.Linear(ff, dim))

        def forward(self, values):
            out, weights = self.attn(values, values, values, need_weights=True, average_attn_weights=False)
            values = self.norm1(values + out)
            values = self.norm2(values + self.ff(values))
            return values, weights

    class Encoder(nn.Module):
        def __init__(self, vocab_size, dim, layers, heads, ff, positions):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, dim, padding_idx=0)
            self.pos = nn.Embedding(positions, dim)
            self.blocks = nn.ModuleList([Block(dim, heads, ff) for _ in range(layers)])
            self.norm = nn.LayerNorm(dim)
            self.head = nn.Linear(dim, 2)

        def forward(self, ids):
            positions = torch.arange(ids.shape[1], device=ids.device).unsqueeze(0)
            values = self.embed(ids) + self.pos(positions)
            attentions = []
            for block in self.blocks:
                values, weights = block(values)
                attentions.append(weights)
            values = self.norm(values)
            logits = self.head(values[:, 0])
            return logits, attentions, values

    return Encoder(
        config["vocab_size"], config["dim"], config["layers"], config["heads"], config["ff"], config["positions"]
    )


def create(config=None, vocab=None, seed=SEED):
    torch = runtime()
    torch.manual_seed(seed)
    tokens = build_vocab() if vocab is None else list(vocab)
    settings = dict(DEFAULT_CONFIG)
    if config:
        settings.update(config)
    settings["vocab_size"] = len(tokens)
    model = build_encoder(torch, settings)
    model.eval()
    return torch, model, Tokenizer(tokens), settings


def save(directory, model, tokenizer, config, extra=None):
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    path = target / LOCAL_FILE
    payload = {
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "vocab": list(tokenizer.vocab),
        "config": dict(config),
        "name": MODEL_NAME,
    }
    if extra:
        payload["extra"] = extra
    torch = runtime()
    torch.save(payload, path)
    return path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(directory):
    target = Path(directory)
    path = target / LOCAL_FILE
    if not path.is_file():
        raise FileNotFoundError(f"веса не найдены: {path}")
    torch = runtime()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    tokenizer = Tokenizer(payload["vocab"])
    config = payload["config"]
    model = build_encoder(torch, config)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return torch, model, tokenizer, config, digest(path)


def encode(tokenizer, segment_text, fragment_text, segment_limit, fragment_limit):
    segment = tokenizer.ids(segment_text, segment_limit)
    fragment = tokenizer.ids(fragment_text, fragment_limit)
    if not segment or not fragment:
        return None
    sequence = [tokenizer.cls_id] + segment + [tokenizer.sep_id] + fragment + [tokenizer.sep_id]
    return {
        "ids": sequence,
        "doc": (1, 1 + len(segment)),
        "ans": (2 + len(segment), 2 + len(segment) + len(fragment)),
    }


def parameters(model):
    return sum(parameter.numel() for parameter in model.parameters())


def config_of(model):
    return {
        "vocab_size": model.embed.num_embeddings,
        "dim": model.embed.embedding_dim,
        "layers": len(model.blocks),
    }


def write_meta(path, payload):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return target
