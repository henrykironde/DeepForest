import os

import pandas as pd
import pytest
import torch

from deepforest import get_data
from deepforest.main import deepforest


@pytest.mark.skipif(
    not os.environ.get("HIPERGATOR"),
    reason="Only run on HIPERGATOR (requires GPU + model downloads).",
)
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available in this test environment.",
)
def test_predict_tile_uses_cuda_when_requested(tmp_path, monkeypatch):
    """Ensure predict_tile runs on CUDA when accelerator/devices request GPU.

    This is a regression test to catch silent CPU fallbacks on GPU nodes.
    """
    log_path = tmp_path / "gpu_benchmark.csv"
    monkeypatch.setenv("DEEPFOREST_BENCHMARK_LOG", str(log_path))

    m = deepforest(config_args={"accelerator": "gpu", "devices": 1, "workers": 0})
    m.load_model(model_name="weecology/deepforest-tree", revision="main")
    m.create_trainer(accelerator="gpu", devices=1)

    results = m.predict_tile(
        path=get_data("OSBS_029.png"),
        patch_size=400,
        patch_overlap=0.0,
        iou_threshold=0.15,
        dataloader_strategy="single",
    )
    assert results is not None and not results.empty

    df = pd.read_csv(log_path)
    assert "pl_device" in df.columns
    assert "trainer_accelerator" in df.columns

    # At least one predict_step should report running on CUDA.
    assert df["pl_device"].astype(str).str.contains("cuda", case=False).any()
