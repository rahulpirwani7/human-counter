# AGENTS.md

This repository is a lightweight human-counting demo that combines a TensorFlow object-detection model with a small React/Vite frontend.

## Project layout

- Root-level Python scripts such as `detect_ssd.py` and `inspect_yolo.py` are model/inference scripts and should be treated as ML experimentation code, not general app logic.
- `model/` contains the TensorFlow SavedModel used by the Python pipeline.
- `yolo26s.pt`, `yolo26s.onnx`, and `yolo26s_saved_model/` are model assets for the YOLO/TFLite side of the project.
- `frontend/` is the web app; it is the main app surface for user interaction and should be edited separately from the ML scripts unless the task explicitly spans both.

## Working conventions

- Prefer small, focused edits that match the current architecture instead of reorganizing the repo.
- Keep model file paths and load names aligned with the scripts that reference them.
- If you change detection logic, check the class threshold and box-processing assumptions used in the Python scripts before shipping.
- Do not remove or rename model assets without updating the code that loads them.
- For frontend changes, stay within the Vite + React pattern already used in `frontend/src`.

## Validation commands

- Frontend build:
  - `cd frontend && npm install`
  - `cd frontend && npm run build`
- Frontend dev server:
  - `cd frontend && npm run dev`
- Python inference scripts:
  - use the repo virtual environments such as `.venv` or `.venv-tf` when running local ML scripts
- This repo does not currently include a formal automated test suite; prefer the smallest relevant validation command for the area you changed.

## Key references

- [readme.md](readme.md)
- [frontend/README.md](frontend/README.md)
- [frontend/package.json](frontend/package.json)

## Guidance for agents

- Treat image processing and object detection as the core domain logic.
- When working on the frontend, keep the app simple and avoid introducing a new app framework or backend unless the task requires it.
- When working on Python ML files, preserve compatibility with the existing TensorFlow/NumPy workflow and avoid broad refactors that hide the model-inference behavior.
