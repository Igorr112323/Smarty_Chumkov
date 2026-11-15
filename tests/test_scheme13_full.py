import tempfile
import unittest
from pathlib import Path

import torch

import tools.train_scheme13_full as training
from tools.preflight_13 import digest


class CheckpointTests(unittest.TestCase):
    def test_optimizer_and_rng_round_trip(self):
        torch.manual_seed(17)
        model = torch.nn.Linear(3, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
        x = torch.tensor([[1.0, 2.0, 3.0]])
        model(x).sum().backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "torch_rng": torch.get_rng_state(), "epoch": 1}
        old_size = training.PART_BYTES
        try:
            training.PART_BYTES = 128
            with tempfile.TemporaryDirectory() as folder:
                names = training.save_state(folder, state, torch)
                self.assertGreater(len(names), 1)
                entries = [{"name": name, "bytes": (Path(folder) / name).stat().st_size, "sha256": digest(Path(folder) / name)} for name in names]
                loaded = training.load_state(folder, entries, torch)
                clone = torch.nn.Linear(3, 1)
                opt2 = torch.optim.AdamW(clone.parameters(), lr=0.01)
                sched2 = torch.optim.lr_scheduler.StepLR(opt2, step_size=1, gamma=0.5)
                clone.load_state_dict(loaded["model"])
                opt2.load_state_dict(loaded["optimizer"])
                sched2.load_state_dict(loaded["scheduler"])
                torch.set_rng_state(loaded["torch_rng"])
                self.assertTrue(torch.equal(torch.get_rng_state(), state["torch_rng"]))
                for current, restored in zip(model.parameters(), clone.parameters()):
                    self.assertTrue(torch.equal(current, restored))
                for net, opt, sched in ((model, optimizer, scheduler), (clone, opt2, sched2)):
                    net(x).sum().backward()
                    opt.step()
                    sched.step()
                    opt.zero_grad()
                for current, restored in zip(model.parameters(), clone.parameters()):
                    self.assertTrue(torch.equal(current, restored))
                self.assertEqual(scheduler.state_dict(), sched2.state_dict())
                (Path(folder) / names[0]).write_bytes(b"broken")
                with self.assertRaises(ValueError):
                    training.load_state(folder, entries, torch)
        finally:
            training.PART_BYTES = old_size


if __name__ == "__main__":
    unittest.main()
