def pytest_sessionstart(session):
    # NumPy/ONNX-only CI deliberately has no torch installed.
    try:
        import torch
    except ImportError:
        return
    torch.set_num_threads(2)
