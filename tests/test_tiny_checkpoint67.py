"""Синтетические проверки загрузки: реальные данные и V5 не нужны."""
import hashlib
import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from spanver import lexfeat
from spanver.tiny_backend import TinyBundle, TinyUnavailable


class Base(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(3))
        self.config = SimpleNamespace()


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = Base()
        self.proj = nn.Linear(3, 2)


def setup_bundle(tmp_path, monkeypatch):
    model = Model()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(0.75)
    checkpoint = tmp_path / 'checkpoint.pt'
    torch.save(model.state_dict(), checkpoint)
    data = checkpoint.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    folder = tmp_path / 'final'; seed = folder / 'seed_0'; seed.mkdir(parents=True)
    (seed / 'part-000.bin').write_bytes(data)
    entry = {'seed': 0, 'sha256': digest, 'size': len(data),
             'chunks': [{'name': 'part-000.bin', 'bytes': len(data), 'sha256': digest}]}
    (folder / 'manifest.json').write_text(json.dumps({'status': 'готово', 'variant': 'v5',
        'restore_verified': True, 'checkpoints': [entry]}), encoding='utf8')
    base_dir = tmp_path / 'base'; base_dir.mkdir()
    head = tmp_path / 'head.json'
    head.write_text(json.dumps({'gate': .45, 'start': .49, 'extend': .245, 'top_chunks': 4,
                                'weights_dir': str(base_dir), 'checkpoint_dir': str(folder),
                                'checkpoint_sha256': digest}), encoding='utf8')
    monkeypatch.setattr(lexfeat, 'load_feature_model', lambda *a, **kw: (object(), Model()))
    return head, folder, model


def test_verified_checkpoint_is_loaded_by_default(tmp_path, monkeypatch):
    head, _, expected = setup_bundle(tmp_path, monkeypatch)
    actual = TinyBundle(head_path=head)
    assert not actual.model.training
    assert all(torch.equal(value, expected.state_dict()[key]) for key, value in actual.model.state_dict().items())


def test_missing_checkpoint_fails_closed(tmp_path, monkeypatch):
    head, folder, _ = setup_bundle(tmp_path, monkeypatch)
    (folder / 'seed_0' / 'part-000.bin').unlink()
    with pytest.raises(TinyUnavailable, match='не загружен'):
        TinyBundle(head_path=head)


def test_corrupted_checkpoint_fails_closed(tmp_path, monkeypatch):
    head, folder, _ = setup_bundle(tmp_path, monkeypatch)
    (folder / 'seed_0' / 'part-000.bin').write_bytes(b'bad')
    with pytest.raises(TinyUnavailable, match='повреждена часть'):
        TinyBundle(head_path=head)


def test_wrong_checkpoint_identity_fails_closed(tmp_path, monkeypatch):
    head, folder, _ = setup_bundle(tmp_path, monkeypatch)
    manifest = json.loads((folder / 'manifest.json').read_text())
    manifest['checkpoints'][0]['sha256'] = '0' * 64
    (folder / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(TinyUnavailable, match='SHA checkpoint'):
        TinyBundle(head_path=head)


def test_checkpoint_part_cannot_escape_seed_dir(tmp_path, monkeypatch):
    head, folder, _ = setup_bundle(tmp_path, monkeypatch)
    manifest = json.loads((folder / 'manifest.json').read_text())
    manifest['checkpoints'][0]['chunks'][0]['name'] = '../checkpoint.pt'
    (folder / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(TinyUnavailable, match='имя части'):
        TinyBundle(head_path=head)
