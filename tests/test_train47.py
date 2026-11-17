import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from tests.test_train46 import Tiny
from tools import train46, train47


class Train47Tests(unittest.TestCase):
    def exercise(self, permanent):
        model = Tiny()
        for parameter in model.parameters():
            torch.nn.init.constant_(parameter, 1.0)
        tokenizer = SimpleNamespace(pad_token_id=0)
        def loss(_, current, __):
            value = sum(parameter.square().sum() for parameter in current.parameters())
            return value, value * 0
        with patch.object(train46, "EPOCHS", 7), patch.object(train46, "PATIENCE", 10), patch.object(train46.lexfeat, "build_table", return_value={"x": 1}), patch.object(train46.lexfeat, "collate", return_value={}), patch.object(train46, "loss_batch", side_effect=loss), patch.object(train46, "validation_loss", side_effect=[7, 6, 5, 4, 3, 2, 1]):
            if permanent is None:
                result = train46.train_variant(model, tokenizer, [1], [2], 0, "v2")
            else:
                result = train46.train_variant(model, tokenizer, [1], [2], 0, "v2", permanent_freeze=permanent)
        return result, model.state_dict()

    def test_v4_is_original_and_v5_freezes_all_epochs(self):
        self.assertEqual(train47.SEEDS, (0, 1, 2, 3, 4))
        old, original = self.exercise(None)
        v4, same = self.exercise(False)
        v5, frozen = self.exercise(True)
        self.assertEqual([h["embeddings_frozen"] for h in v4["history"]], [True] * 5 + [False] * 2)
        self.assertEqual([h["embeddings_frozen"] for h in v5["history"]], [True] * 7)
        self.assertEqual([h["train_loss"] for h in old["history"]], [h["train_loss"] for h in v4["history"]])
        for key in original:
            self.assertTrue(torch.equal(original[key], same[key]))
        key = "base.bert.embeddings.weight"
        self.assertFalse(torch.equal(same[key], frozen[key]))


if __name__ == "__main__":
    unittest.main()
