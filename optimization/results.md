# Edge optimization results

Measured on **Apple M4**, 32 GB RAM, macOS-26.6.2-arm64-arm-64bit-Mach-O, torch 2.14.1 (4 CPU threads). GPU = Apple M4 GPU (MPS / CoreML).
Batch 1, 640x640 input, 20 warm-up runs then the average over 200 runs. FPS is model inference only. mAP is on the held-out test split for every row.

| Variant | Precision | Size (MB) | CPU FPS | GPU FPS | Neural Engine FPS | mAP@50 | mAP@50-95 |
|---|---|---|---|---|---|---|---|
| Baseline (PyTorch) | FP32 | 5.5 | 44.2 | 163.0 | – | 0.853 | 0.607 |
| CoreML FP16 | FP16 | 5.4 | 109.2 | 223.0 | 634.4 | 0.845 | 0.586 |
| ONNX Runtime INT8 | INT8 weights + activations | 3.3 | 71.3 | – | – | 0.824 | 0.576 |
| Pruned 30% | FP32, 30% of conv weights zeroed | 5.5 | 41.2 | 160.5 | – | 0.848 | 0.605 |

Change vs. the FP32 PyTorch baseline:

| Variant | Size | Best CPU FPS | Best GPU-side FPS | mAP@50 |
|---|---|---|---|---|
| CoreML FP16 | 0.99x | 2.47x | 3.89x | -0.009 |
| ONNX Runtime INT8 | 0.60x | 1.61x | – | -0.029 |
| Pruned 30% | 1.00x | 0.93x | 0.98x | -0.006 |
