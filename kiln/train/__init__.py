"""`kiln train`: train a detection model from a config file, then export it to ONNX.

The steps are the same for every training library (config → dataset → train → export).
Only the library-specific part lives in `kiln/train/backends/`, one file per library.
"""
