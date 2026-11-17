import unittest
from unittest.mock import patch

import torch
from torch import nn

from spanver.decode44 import predict44
from tools.nested_cv35 import predict
from tools.train44 import build_optimizer, clone_state, input_without_gold, set_embedding_epoch, train, update_ema


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Module()
        self.base.bert = nn.Module()
        self.base.bert.embeddings = nn.Linear(2, 2)
        self.base.bert.encoder = nn.Linear(2, 2)
        self.base.classifier = nn.Linear(2, 2)
        self.proj = nn.Linear(2, 2)

    def forward(self, value):
        return self.base.classifier(self.base.bert.encoder(self.proj(self.base.bert.embeddings(value))))


class Train44Tests(unittest.TestCase):
    def test_embedding_freeze_and_lr_group(self):
        model = ToyModel()
        optimizer, _, embeddings, warmup = build_optimizer(model, 100)
        self.assertEqual(warmup, 10)
        self.assertAlmostEqual(optimizer.param_groups[0]["initial_lr"], optimizer.param_groups[1]["initial_lr"] * 0.1)
        set_embedding_epoch(embeddings, 5)
        self.assertTrue(all(not value.requires_grad for value in embeddings))
        set_embedding_epoch(embeddings, 6)
        self.assertTrue(all(value.requires_grad for value in embeddings))

    def test_ema_updates_values(self):
        model = ToyModel()
        ema = clone_state(model)
        key = next(iter(ema))
        before = ema[key].clone()
        with torch.no_grad():
            for value in model.parameters():
                value.add_(1.0)
        update_ema(ema, model, 0.999)
        self.assertTrue(torch.allclose(ema[key], before + 0.001, atol=1e-6))

    def test_synthetic_training_runs_through_unfreeze(self):
        model = ToyModel()
        tokenizer = type("Tokenizer", (), {"pad_token_id": 0})()
        pairs = [{"pair_id": "a"}, {"pair_id": "b"}]
        table = {row["pair_id"]: row for row in pairs}
        def fake_loss(torch_module, current, piece):
            output = current(piece)
            return output.square().mean(), output.mean().square()
        with patch("tools.train44.EPOCHS", 6), patch("tools.train44.BATCH", 2), patch("tools.train44.MICRO", 1), patch("tools.train44.lexfeat.build_table", return_value=table), patch("tools.train44.lexfeat.collate", side_effect=lambda rows, pad: torch.ones((len(rows), 2))), patch("tools.train44.loss_batch", side_effect=fake_loss):
            result = train(model, tokenizer, pairs, pairs, 0)
        self.assertEqual(result["epochs_ran"], 6)
        self.assertTrue(result["history"][4]["embeddings_frozen"])
        self.assertFalse(result["history"][5]["embeddings_frozen"])
        self.assertGreaterEqual(result["best_epoch"], 1)

    def test_blinding_removes_gold(self):
        self.assertEqual(input_without_gold({"pair_id": "p", "spans": [{"start": 0}], "label": 1, "document_text": "abc"}), {"pair_id": "p", "document_text": "abc"})

    def test_baseline_decoder_matches_existing(self):
        value = {"answer": "Текст. Другое!", "pieces": [[0, 5, 0.7], [7, 13, 0.2]]}
        self.assertEqual(predict44(value, 0.5, 0.25), predict(value, 0.5))


if __name__ == "__main__":
    unittest.main()
