import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from tools import train46
from tools.predict_frozen38 import load_state
from spanver.stage_train import ENCODER_RATE, HEAD_RATE


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base = torch.nn.Module()
        self.base.bert = torch.nn.Module()
        self.base.bert.embeddings = torch.nn.Linear(1, 1)
        self.base.bert.encoder = torch.nn.Linear(1, 1)
        self.base.classifier = torch.nn.Linear(1, 1)
        self.proj = torch.nn.Linear(1, 1)


class Train46Tests(unittest.TestCase):
    def test_only_v2_v3_reduce_embedding_lr(self):
        model = Tiny()
        normal, embeddings = train46.groups(model, False)
        frozen, other = train46.groups(model, True)
        self.assertEqual([g["lr"] for g in normal], [ENCODER_RATE, HEAD_RATE])
        self.assertEqual([g["lr"] for g in frozen], [ENCODER_RATE * 0.1, ENCODER_RATE, HEAD_RATE])
        self.assertEqual(len(embeddings), len(other))

    def test_ema(self):
        model = Tiny()
        state = train46.clone_state(model)
        before = {key: value.clone() for key, value in state.items()}
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(1)
        train46.update_ema(state, model)
        for name in state:
            self.assertTrue(torch.allclose(state[name], before[name] + 0.001, atol=1e-6))

    def test_chunks_and_restore(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Tiny()
            with patch.object(train46, "ROOT", Path(directory)), patch.object(train46, "PART_BYTES", 128):
                entry = train46.save_checkpoint(model, "v1", 1, 0)
            folder = Path(directory) / "weights/ablation46/v1/fold1/seed_0"
            data = b"".join((folder / item["name"]).read_bytes() for item in entry["chunks"])
            self.assertTrue(all(item["bytes"] <= 128 for item in entry["chunks"]))
            self.assertEqual(hashlib.sha256(data).hexdigest(), entry["sha256"])
            restored = Tiny()
            load_state(torch, restored, folder, entry)
            for key, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, restored.state_dict()[key]))


if __name__ == "__main__":
    unittest.main()
