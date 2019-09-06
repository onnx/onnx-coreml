---
name: "\U0001F41E Bug report"
about: Submit a bug report
title: ''
labels: bug
assignees: ''

---

## 🐞Describe the bug
A clear and brief description of what the bug is.

## Trace
If applicable, please paste the error trace.

## To Reproduce
- If a python script can reproduce the error, please paste the code snippet
```
from onnx_coreml import convert
# Paste code snippet here
```
- If applicable, please attach ONNX model
    - If model cannot be shared publicly, please attach it via filing a bug report at https://developer.apple.com/bug-reporting/ 
- If model conversion succeeds, however, there is numerical mismatch between the original and the coreml model, please paste python script used for comparison (pytorch code, onnx runtime code etc.)

## System environment (please complete the following information):
 - coremltools version  (e.g., 3.0b5):
 - onnx-coreml version (e.g. 1.0b2):
 - OS (e.g., MacOS, Linux):
 - macOS version (if applicable):
 - How you install python (anaconda, virtualenv, system):
 - python version (e.g. 3.7):
 - any other relevant information:

## Additional context
Add any other context about the problem here.
