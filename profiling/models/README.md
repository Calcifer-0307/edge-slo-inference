# Local Model Artifacts

This directory is intentionally kept out of version control.

The ONNX model files used for profiling are generated locally and should not be committed to the repository because they are large binary artifacts.

To recreate the models locally, run:

```bash
python3 profiling/generate_onnx.py
```

This will download the pretrained torchvision weights on demand and export the following files into this directory:

- `mobilenet_v2.onnx`
- `resnet18.onnx`
- `resnet50.onnx`

Then run:

```bash
python3 profiling/profile_models.py
```

to regenerate `data/service_time.csv`.
