"""V21 шаг 2: архитектура cross-encoder для поиска неподтверждённых фрагментов.

`CrossEncoderSpan`: rubert-tiny2 (реальные локальные веса, §4.2) + две головы:

  * span-голова — логиты старта/конца по токенам кандидата (BCE по каждому токену, метка — gold-span);
  * риск-голова — логит «в этом кандидате есть галлюцинация» (CLS-пулинг).

Вход: `[CLS] document_prefix [SEP] candidate_tokens [SEP]`, `token_type_ids` разделяют документ
и кандидата. Документ режется до `doc_tokens` (кэшируется на документ), кандидат подаётся целиком.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
from torch import nn
from transformers import BertConfig, BertModel, BertTokenizerFast

# v22: энкодер настраивается (по умолчанию — прежний rubert-tiny2), чтобы тот же CE можно было
# обучить на rubert-base-cased, не меняя архитектуру и не ломая воспроизводимость v21
MODEL_DIR = Path(os.environ.get("CE_MODEL_DIR") or (ROOT / "models/rubert-tiny2"))


class CrossEncoderSpan(nn.Module):
    def __init__(self, encoder: nn.Module, dropout: float = 0.1):
        super().__init__()
        self.encoder = encoder
        hidden = encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.risk = nn.Linear(hidden, 1)
        self.start = nn.Linear(hidden, 1)
        self.end = nn.Linear(hidden, 1)

    def forward(self, input_ids, attention_mask, token_type_ids, candidate_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                               token_type_ids=token_type_ids)
        hidden = self.dropout(outputs.last_hidden_state)
        risk_logit = self.risk(hidden[:, 0]).squeeze(-1)
        start_logits = self.start(hidden).squeeze(-1)
        end_logits = self.end(hidden).squeeze(-1)
        mask = candidate_mask.bool()
        start_logits = start_logits.masked_fill(~mask, -1e4)
        end_logits = end_logits.masked_fill(~mask, -1e4)
        return {"risk_logit": risk_logit, "start_logits": start_logits, "end_logits": end_logits}


def load_tokenizer(model_dir=None):
    return BertTokenizerFast.from_pretrained(str(Path(model_dir or MODEL_DIR)))


def build_model(use_pretrained=True, seed=21, model_dir=None):
    """Собирает модель на реальных весах энкодера (по умолчанию rubert-tiny2, сверен по SHA)."""
    model_dir = Path(model_dir or MODEL_DIR)
    config = BertConfig.from_pretrained(str(model_dir))
    if use_pretrained and (model_dir / "pytorch_model.bin").exists():
        encoder = BertModel.from_pretrained(str(model_dir), config=config)
    else:
        torch.manual_seed(seed)
        encoder = BertModel(config)
    return CrossEncoderSpan(encoder)


def count_parameters(model):
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {"total": total, "trainable": trainable,
            "fp32_mb": round(total * 4 / 1024 ** 2, 2),
            "int8_mb": round(total * 1 / 1024 ** 2, 2)}


def encode_batch(tokenizer, examples, doc_tokens=200, max_len=384):
    """Готовит батч по списку (document, candidate_text) с метками спанов внутри кандидата."""
    input_ids, attention_mask, token_type_ids, candidate_mask = [], [], [], []
    start_targets, end_targets, risk_targets = [], [], []
    for document, candidate, span, risk in examples:
        doc_ids = tokenizer(document, add_special_tokens=False, truncation=True,
                            max_length=doc_tokens)["input_ids"]
        cand = tokenizer(candidate, add_special_tokens=False, truncation=True,
                         max_length=max_len - doc_tokens - 3)
        cand_ids = cand["input_ids"]
        ids = [tokenizer.cls_token_id] + doc_ids + [tokenizer.sep_token_id] + cand_ids + \
              [tokenizer.sep_token_id]
        types = [0] * (len(doc_ids) + 2) + [1] * (len(cand_ids) + 1)
        mask = [1] * len(ids)
        cand_mask = [0] * (len(doc_ids) + 2) + [1] * len(cand_ids) + [0]
        start = end = -100
        if span is not None and cand_ids:
            offsets = tokenizer(candidate, add_special_tokens=False, return_offsets_mapping=True,
                                truncation=True, max_length=len(cand_ids))["offset_mapping"]
            base = len(doc_ids) + 2
            start_index = next((index for index, (left, right) in enumerate(offsets)
                                if right > span[0]), None)
            end_index = next((index for index in range(len(offsets) - 1, -1, -1)
                              if offsets[index][0] < span[1]), None)
            if start_index is not None and end_index is not None and start_index <= end_index:
                start = base + start_index
                end = base + end_index
        length = max_len
        input_ids.append(ids + [tokenizer.pad_token_id] * (length - len(ids)))
        attention_mask.append(mask + [0] * (length - len(mask)))
        token_type_ids.append(types + [0] * (length - len(types)))
        candidate_mask.append(cand_mask + [0] * (length - len(cand_mask)))
        start_targets.append(start)
        end_targets.append(end)
        risk_targets.append(risk)

    batch = {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
        "candidate_mask": torch.tensor(candidate_mask, dtype=torch.long),
    }
    return batch, torch.tensor(start_targets), torch.tensor(end_targets), \
        torch.tensor(risk_targets, dtype=torch.float32)
