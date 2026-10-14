# Edge optimization results

Measured on **Apple M4**, 32 GB RAM, macOS-26.6.2-arm64-arm-64bit-Mach-O, torch 2.14.1 (4 CPU threads). GPU = Apple M4 GPU (MPS / CoreML).
Batch 1, 640x640 input, 20 warm-up runs then the average over 200 runs. FPS is model inference only. mAP is on the held-out test split for every row.

| Variant | Precision | Size (MB) | CPU FPS | GPU FPS | Neural Engine FPS | mAP@50 | mAP@50-95 |
|---|---|---|---|---|---|---|---|
| Baseline (PyTorch) | FP32 | 5.5 | 44.2 | 163.0 | – | 0.853 | 0.607 |
| PyTorch half | FP16 | 5.5 | – | 198.6 | – | 0.854 | 0.607 |
| ONNX Runtime | FP32 | 10.7 | 66.4 | – | – | 0.850 | 0.587 |
| ONNX Runtime INT8 | INT8 weights + activations | 3.3 | 71.3 | – | – | 0.824 | 0.576 |
| ONNX Runtime INT8, conv only | INT8 convs, float everything else | 3.2 | 56.3 | – | – | 0.831 | 0.577 |
| CoreML FP16 | FP16 | 5.4 | 109.2 | 223.0 | 634.4 | 0.845 | 0.586 |
| CoreML INT8 | INT8 weights | 2.9 | 109.7 | 62.6 | 637.5 | 0.841 | 0.585 |
| Pruned 30% | FP32, 30% conv weights zeroed | 5.5 | 41.2 | 160.5 | – | 0.848 | 0.605 |
| Pruned 50% | FP32, 50% conv weights zeroed | 5.5 | 41.1 | 160.3 | – | 0.766 | 0.525 |

Change vs. the FP32 PyTorch baseline:

| Variant | Size | Best CPU FPS | Best GPU-side FPS | mAP@50 |
|---|---|---|---|---|
| PyTorch half | 1.00x | – | 1.22x | +0.001 |
| ONNX Runtime | 1.94x | 1.50x | – | -0.004 |
| ONNX Runtime INT8 | 0.60x | 1.61x | – | -0.029 |
| ONNX Runtime INT8, conv only | 0.58x | 1.27x | – | -0.023 |
| CoreML FP16 | 0.99x | 2.47x | 3.89x | -0.009 |
| CoreML INT8 | 0.52x | 2.48x | 3.91x | -0.012 |
| Pruned 30% | 1.00x | 0.93x | 0.98x | -0.006 |
| Pruned 50% | 1.00x | 0.93x | 0.98x | -0.087 |

Notes from the run:

- PyTorch half: FP16 on CPU isn't a real option in PyTorch, GPU only
