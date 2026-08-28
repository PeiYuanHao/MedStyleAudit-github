from medstyleaudit.data.loading import loader_kwargs


def test_loader_kwargs_enable_cuda_prefetch_only_with_workers():
    settings = {"batch_size": 128, "num_workers": 8, "prefetch_factor": 3}
    kwargs = loader_kwargs(settings, "cuda:0")
    assert kwargs == {
        "batch_size": 128,
        "num_workers": 8,
        "pin_memory": True,
        "persistent_workers": True,
        "prefetch_factor": 3,
    }


def test_loader_kwargs_omit_worker_only_options_for_single_process():
    kwargs = loader_kwargs({"batch_size": 4, "num_workers": 0}, "cpu")
    assert kwargs == {"batch_size": 4, "num_workers": 0, "pin_memory": False}
