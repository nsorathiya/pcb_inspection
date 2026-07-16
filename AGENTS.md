# AGENTS.md - PCB AOI Project Instructions

## Goal
Build a file-based PCB AOI MVP using paired 2D images and 3D height/depth maps. The system must return PASS, FAIL, or UNCERTAIN and store inspection history, confidence, model version, recipe version, timestamp, processing time, and audit information.

## Current scope
In scope:
- File upload only
- Mock 3D/reference data until real data arrives
- FastAPI backend
- SQLite
- React + Vite
- Dataset validation
- 2D/3D preprocessing interfaces
- Replaceable inference layer
- Mock inference first
- ONNX Runtime later
- Tests and Windows local setup

Out of scope for now:
- Cameras
- PLC
- Hardware trigger
- Conveyor
- MES/ERP
- HALCON as core dependency

## Rules
1. Work phase by phase.
2. Inspect before editing.
3. Keep diffs small.
4. Do not delete working code without explicit approval.
5. Preserve raw uploads unchanged.
6. Keep runtime data outside Git-tracked source folders.
7. Never silently convert 16-bit/32-bit height data to 8-bit.
8. Treat 2D and 3D as a strict pair.
9. Invalid or mismatched inputs must never return PASS.
10. Use a replaceable inference interface:
   - MockInferenceEngine first
   - OnnxInferenceEngine later
11. Add tests for meaningful changes.
12. Run tests before reporting completion.
13. Do not fabricate accuracy.
14. Do not train a real model before verified labelled data arrives.
15. Record model and recipe version for every inspection.

## Preferred stack
Python 3.12+, FastAPI, Pydantic, SQLAlchemy/SQLModel, SQLite WAL, OpenCV, NumPy, PyTorch, ONNX Runtime, React, Vite, TypeScript, Pytest, Playwright.

## Initial API
- GET /api/v1/health
- POST /api/v1/inspections
- GET /api/v1/inspections
- GET /api/v1/inspections/{id}
- GET /api/v1/recipes
- POST /api/v1/recipes
- GET /api/v1/models
- POST /api/v1/models/activate

## Definition of done
A task is complete only when:
- Acceptance criteria are met
- Tests are added or updated
- Tests pass
- Files changed are summarized
- Limitations are listed
- No unrelated refactor is included

## Codex completion report
1. What you inspected
2. What you changed
3. Files changed
4. Tests run and results
5. Risks/limitations
6. Recommended next task
