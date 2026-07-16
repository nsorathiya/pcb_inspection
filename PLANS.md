# PLANS.md - PCB AOI File-Based MVP

## Strategy
Build the software platform before real vision data arrives. Use mock inference and synthetic/reference files. Replace mock inference after verified paired 2D/3D data is received.

Do not execute all phases at once.

## Phase 0 - Repository assessment
Deliver:
- docs/repository_assessment.md
- docs/gap_analysis.md
- docs/proposed_phase_1_changes.md

No feature implementation.

## Phase 1 - Foundation
- Clean repository structure
- Backend/frontend/ml/contracts/runtime folders
- Environment config
- Logging
- .gitignore
- Health endpoint
- Baseline tests

## Phase 2 - Data contracts and validation
- metadata.schema.json
- dataset_contract.md
- 2D/3D pair validator
- Dataset audit CLI
- Corruption, format, dimension, bit-depth checks

## Phase 3 - Database and storage
- SQLite WAL
- Inspection, recipe, model, audit tables
- Immutable raw-file storage
- Inspection APIs

## Phase 4 - Preprocessing
- 2D loader
- Native-depth 3D loader
- ROI extraction
- Normalization interfaces
- Preview generation
- Deterministic tests

## Phase 5 - Replaceable inference
- InferenceEngine interface
- MockInferenceEngine
- PASS/FAIL/UNCERTAIN
- Model registry
- Threshold configuration
- OnnxInferenceEngine shell

## Phase 6 - Operator UI
- Upload 2D + 3D
- Recipe selection
- Result display
- Confidence and latency
- 2D/3D previews
- History
- Validation errors

## Phase 7 - Engineer mode
- Recipes
- Models
- Thresholds
- Raw/normalized previews
- Review/correction workflow
- Audit trail

## Phase 8 - Real data integration
Only after verified data arrives:
- Dataset audit
- Leakage-safe split
- 2D-only baseline
- 3D-only baseline
- Fusion model
- Precision, recall, F1, false-pass, false-fail, confusion matrix

## Phase 9 - ONNX
- Export
- Parity test
- Production inference
- Latency benchmark
- Model activation

## Phase 10 - Stabilization
- Regression tests
- Backup/restore
- Windows scripts
- Documentation
- Release checklist
