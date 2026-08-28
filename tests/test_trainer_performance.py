import pytest

torch = pytest.importorskip("torch")

from medstyleaudit.models.trainer import train


def test_amp_configuration_falls_back_to_fp32_on_cpu(tmp_path):
    images = torch.rand(8, 3, 8, 8)
    labels = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
    identifiers = torch.arange(8)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(images, labels, identifiers), batch_size=4)
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 8 * 8, 1))
    config = {
        "training": {
            "epochs": 1,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "selection_metric": "val_auroc",
            "maximize_metric": True,
            "save_predictions": False,
            "amp": True,
            "channels_last": True,
        }
    }
    history = train(model, loader, loader, config, tmp_path, device="cpu")
    checkpoint = torch.load(tmp_path / "last.ckpt", map_location="cpu", weights_only=False)
    assert len(history) == 1
    assert checkpoint["precision"] == "fp32"
