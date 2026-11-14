from __future__ import annotations

from dataclasses import dataclass, field

Lang = str
LANG_TO_PASSAGE = {"en": "Passage", "ru": "Фрагмент"}
TEMPLATE = {
    "ru": "Контекст:\n{context}\n\nВопрос: {question}\n\nОтвет:",
    "en": "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:",
}


class PromptUtils:
    @staticmethod
    def format_context(context, question, lang):
        word = LANG_TO_PASSAGE.get(lang, LANG_TO_PASSAGE["en"])
        lines = [f"{word} {index + 1}: {passage}" for index, passage in enumerate(context)]
        body = "\n".join(lines)
        template = TEMPLATE.get(lang, TEMPLATE["en"])
        return template.format(context=body, question=question or "")


@dataclass
class HallucinationSample:
    prompt: str = ""
    answer: str = ""
    labels: list = field(default_factory=list)
    split: str = ""
    dataset: str = ""
    language: str = ""
    answer_start: int = 0
    gold_spans: list = field(default_factory=list)


class HallucinationDetector:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def predict(self, context, question, answer, output_format="spans"):
        raise NotImplementedError("детектор-обёртка не переносится, используйте TransformerDetector")


class HallucinationDataset:
    @classmethod
    def prepare_tokenized_input(cls, tokenizer, context, answer, max_length=4096):
        import torch

        encoding = tokenizer(
            context,
            answer,
            truncation="only_first",
            max_length=max_length,
            return_offsets_mapping=True,
            return_tensors="pt",
            add_special_tokens=True,
        )
        offsets = encoding.pop("offset_mapping")[0]
        answer_only = tokenizer(answer, add_special_tokens=False, return_tensors="pt")
        answer_token_count = answer_only["input_ids"].shape[1]
        total_seq_len = encoding["input_ids"].shape[1]
        answer_start_token = max(0, total_seq_len - answer_token_count - 1)
        labels = [-100] * answer_start_token + [0] * (total_seq_len - answer_start_token)
        labels = labels[:total_seq_len]
        return encoding, labels, offsets, answer_start_token

    @classmethod
    def collate(cls, samples, tokenizer, max_length=4096):
        prompts = [sample.prompt for sample in samples]
        answers = [sample.answer for sample in samples]
        batch = tokenizer(
            prompts,
            answers,
            truncation="only_first",
            max_length=max_length,
            padding=True,
            return_offsets_mapping=True,
            return_tensors="pt",
            add_special_tokens=True,
        )
        offsets = batch.pop("offset_mapping")
        labels = batch["input_ids"].clone()
        labels[:, :] = -100
        for index in range(len(samples)):
            length = int(batch["attention_mask"][index].sum().item())
            answer_tokens = tokenizer(answers[index], add_special_tokens=False, return_tensors="pt")["input_ids"].shape[1]
            start = max(0, length - answer_tokens - 1)
            labels[index, start:length] = 0
        return {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"], "labels": labels}, offsets


def sklearn_metrics():
    try:
        from sklearn.metrics import auc, classification_report, precision_recall_fscore_support, roc_curve

        return auc, classification_report, precision_recall_fscore_support, roc_curve
    except ImportError as error:
        raise RuntimeError("sklearn не установлен; метрики LettuceDetect не используются") from error
