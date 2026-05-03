"""Tests for utils/model_persistence.py"""

import json

import torch

from utils.model_persistence import MANIFEST_FILENAME, _state_dict_hash, save_model


def _make_model(in_dim=4, out_dim=1):
    return torch.nn.Linear(in_dim, out_dim)


class TestStateDictHash:
    def test_deterministic(self):
        model = _make_model()
        sd = model.state_dict()
        assert _state_dict_hash(sd) == _state_dict_hash(sd)

    def test_different_weights_different_hash(self):
        m1 = _make_model()
        m2 = _make_model()
        torch.nn.init.ones_(m2.weight)
        assert _state_dict_hash(m1.state_dict()) != _state_dict_hash(m2.state_dict())

    def test_returns_8_hex_chars(self):
        h = _state_dict_hash(_make_model().state_dict())
        assert len(h) == 8
        int(h, 16)  # should not raise


class TestSaveModel:
    def test_creates_canonical_and_archive(self, tmp_path):
        model = _make_model()
        record = save_model(model, "mlp", tmp_path)

        canonical = tmp_path / "mlp.pth"
        archive = tmp_path / record["archive_path"]
        assert canonical.exists()
        assert archive.exists()

    def test_record_fields(self, tmp_path):
        model = _make_model()
        record = save_model(model, "test_net", tmp_path)

        assert record["name"] == "test_net"
        assert "hash" in record
        assert "timestamp" in record
        assert record["canonical_path"] == "test_net.pth"
        assert "archive_path" in record

    def test_manifest_appended(self, tmp_path):
        model = _make_model()
        save_model(model, "m1", tmp_path)
        save_model(model, "m2", tmp_path)

        manifest = tmp_path / "models" / MANIFEST_FILENAME
        assert manifest.exists()
        history = json.loads(manifest.read_text())
        assert len(history) == 2
        assert history[0]["name"] == "m1"
        assert history[1]["name"] == "m2"

    def test_metadata_stored(self, tmp_path):
        model = _make_model()
        record = save_model(model, "net", tmp_path, metadata={"epochs": 10, "mse": 0.5})
        assert record["metadata"]["epochs"] == 10

    def test_roundtrip(self, tmp_path):
        model = _make_model()
        save_model(model, "rt", tmp_path)

        loaded_state = torch.load(tmp_path / "rt.pth", weights_only=True)
        model2 = _make_model()
        model2.load_state_dict(loaded_state)

        x = torch.randn(1, 4)
        assert torch.allclose(model(x), model2(x))

    def test_corrupt_manifest_recovered(self, tmp_path):
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        (models_dir / MANIFEST_FILENAME).write_text("not json")

        model = _make_model()
        save_model(model, "fix", tmp_path)

        history = json.loads((models_dir / MANIFEST_FILENAME).read_text())
        assert len(history) == 1
