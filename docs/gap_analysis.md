# Gap Analysis

Date: 2026-07-16

## Target

The target is a file-based industrial PCB AOI MVP that accepts an aligned 2D image and native-depth 3D height map, validates them as a strict pair, preserves originals, returns PASS/FAIL/UNCERTAIN, and records enough context to reproduce and audit every decision.

## Priority definitions

- P0: blocks trustworthy operation or can produce an incorrect decision.
- P1: required for the file-based MVP.
- P2: required before production deployment or real-model activation.
- P3: later integration or optimization.

## Functional and technical gaps

| Area | Current state | Required state | Priority | Recommended phase |
| --- | --- | --- | --- | --- |
| Decision safety | Four defect labels return as an annotated image. | Structured PASS/FAIL/UNCERTAIN; invalid and low-confidence inputs can never silently pass. | P0 | Phase 5 |
| Label correctness | Training and inference class order disagree. | Persisted, versioned class map shared by training and inference, with tests. | P0 | Phase 8 before any real-model use |
| Evaluation integrity | Test images duplicate training images. | Grouped, leakage-safe train/validation/test split with immutable manifest. | P0 | Phase 8 |
| 2D/3D input | One 2D upload only. | Strict 2D image + 3D height/depth + metadata contract. | P0 | Phase 2 |
| Depth preservation | Upload is renamed `.jpg`; default image loading is 8-bit/color oriented. | Preserve raw bytes and load depth with unchanged native dtype and documented units. | P0 | Phases 2 and 4 |
| Input validation | No content, decode, size, dimension, depth, or alignment checks. | Layered validation with explicit machine-readable errors. | P0 | Phase 2 |
| Model evidence | Small checkpoint has no metrics or provenance. | Versioned model package with manifest, class map, data version, metrics, thresholds, and hash. | P0 | Phases 8 and 9 |
| Application startup | Depends on current directory and immediately loads a model. | Stable package entry point, explicit settings, lifecycle handling, and model-independent health check. | P1 | Phase 1 |
| API versioning | Only `POST /predict`. | Versioned `/api/v1` contract with health, inspections, recipes, and models. | P1 | Phases 1, 3, and 5 |
| Configuration | URLs, paths, and CORS are hard-coded. | Typed environment configuration with safe defaults and example file. | P1 | Phase 1 |
| Logging | Print statements/no request context. | Structured logging with request/inspection IDs and no sensitive/raw data in logs. | P1 | Phase 1 |
| Runtime storage | Uploads and outputs are tracked under source directories. | Configurable untracked runtime root with immutable originals and derived artifacts. | P1 | Phases 1 and 3 |
| Persistence | None. | SQLite WAL with inspection, recipe, model, and audit records. | P1 | Phase 3 |
| Inference architecture | PyTorch model directly embedded in request flow. | Replaceable inference interface; deterministic mock first, ONNX later. | P1 | Phase 5 |
| Preprocessing | Resize and tensor conversion only. | Separate 2D/native-depth 3D loaders, registration checks, ROI, normalization, and preview paths. | P1 | Phase 4 |
| Frontend workflow | One-file upload and image result. | Paired input, metadata, result state, validation errors, previews, and history. | P1 | Phase 6 |
| Auditability | No inspection record or versions. | Store hashes, versions, timestamps, latency, decision, confidence, inputs, and corrections. | P1 | Phases 3, 5, and 7 |
| Automated tests | None. | Backend unit/API tests, frontend tests, end-to-end critical path, and deterministic fixtures. | P1 | Begin in Phase 1; expand every phase |
| Dependency control | Unpinned Python packages; missing local environments. | Declared supported versions and reproducible lock/constraint strategy. | P1 | Phase 1 |
| Git hygiene | Generated files, datasets, caches, and model artifacts are tracked. | Root ignore rules and documented artifact/data policy; reviewed migration for existing tracked files. | P1 | Phase 1 proposal, separate approval for untracking |
| Security | Wildcard CORS and no authentication/limits. | Restricted origins, upload limits, safe filenames, secure error handling; auth based on deployment needs. | P2 | Phases 1, 2, and 10 |
| Database concurrency | No database. | SQLite WAL, busy timeout, bounded transactions, backup/restore verification. | P2 | Phases 3 and 10 |
| Model deployment | PyTorch checkpoint loaded directly. | ONNX export, numerical parity test, latency benchmark, atomic activation and rollback. | P2 | Phase 9 |
| Monitoring | None. | Health/readiness, latency and error metrics, disk monitoring, model/drift review process. | P2 | Phase 10 |
| Packaging | Manual commands only. | Tested Windows startup scripts, release checklist, versioned configuration, backup instructions. | P2 | Phase 10 |
| Camera/PLC/MES | None; intentionally out of current scope. | Vendor SDK adapters and industrial handshakes after file MVP is stable. | P3 | Future phase |

## Highest-risk findings

1. The current label-index mismatch can report the wrong defect class.
2. The nominal test set is fully leaked into training, so it cannot measure generalization.
3. Current upload handling cannot preserve or validate industrial 3D data.
4. Invalid inputs have no fail-safe decision path.
5. The checkpoint has no traceable class mapping, dataset version, metrics, or threshold provenance.
6. Runtime artifacts and datasets are mixed with source control, increasing accidental data exposure and repository growth.

## What can be reused

- FastAPI and React/Vite remain appropriate framework choices.
- Existing 2D prototype code can serve as a behavior/reference fixture during migration.
- OpenCV, PyTorch, and the current images are useful for experiments after data governance is established.
- The current endpoint can temporarily remain available while the versioned foundation is added.

## What should not be trusted as production evidence

- Current prediction labels.
- Current model accuracy or generalization.
- The `backend/test` folder as an independent test set.
- The current checkpoint as a deployable industrial model.
- The existing upload path for 3D height/depth data.

## Recommended order

Proceed in the order defined by `PLANS.md`. The first implementation should be limited to a backend foundation with deterministic startup, typed settings, logging, root Git hygiene for future artifacts, `GET /api/v1/health`, and baseline tests. Do not introduce paired inspection behavior until the data contract is reviewed in Phase 2.
