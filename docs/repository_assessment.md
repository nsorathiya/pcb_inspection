# Repository Assessment

Date: 2026-07-16

## Scope

This assessment covers the current `AOI_FOR_PCB` repository at commit `b720c31`. It is inspection and planning only. No application feature, model, dataset, or existing source file was changed.

## Executive summary

The repository contains a small proof-of-concept PCB image classifier:

- A FastAPI endpoint accepts one image and returns an annotated image.
- A custom PyTorch CNN classifies four 2D defect labels.
- A React/Vite page uploads one image to a hard-coded local endpoint.
- A 32-image grayscale dataset, a saved model, and generated runtime files are committed to Git.

It is not yet a 2D/3D AOI platform. There is no paired-input contract, native-depth loading, inspection database, audit trail, recipe/model registry, PASS/FAIL/UNCERTAIN decision contract, health endpoint, automated test suite, or production configuration.

The existing prototype should be preserved as reference code while the foundation is introduced incrementally. It should not be treated as a validated model: the dataset is very small and imbalanced, the nominal test images duplicate training images, and training/inference class ordering is inconsistent.

## Repository map

| Area | Current contents | Assessment |
| --- | --- | --- |
| Root | `README.md` | Prototype setup notes; paths and runtime assumptions need revision. |
| Backend entry points | `backend/main.py`, `backend/api.py` | Two near-duplicate FastAPI applications expose `POST /predict`. |
| Inference | `backend/predictor.py` | Loads a PyTorch checkpoint during module import and performs single-image 2D classification. |
| Model | `backend/model.py` | Small custom CNN for 224 x 224, three-channel inputs. |
| Training | `backend/trainer.py`, `backend/data_loader.py` | Trains on the complete `ImageFolder`; no validation split, metrics, seed, checkpoint metadata, or leakage control. |
| Visualization | `backend/visualize_neurons.py` | Interactive Matplotlib utility for saved activation tensors. Matplotlib is not declared in requirements. |
| Frontend | `frontend/src/*` | Single-file JavaScript React upload screen. It is not TypeScript and supports only one 2D image. |
| Dataset | `backend/dataset`, `backend/dataset_raw`, `backend/test` | All current images are 2D grayscale; content is duplicated across folders. |
| Runtime artifacts | `backend/uploads`, `backend/annotated_output`, `backend/neuron_data` | Generated files are committed to Git and written relative to the process working directory. |
| Model artifact | `backend/saved_model/best_model.pth` | A 51.4 MB checkpoint is committed without version, metrics, training manifest, or provenance. |
| Tests | None | No Python or frontend automated tests are configured. `backend/test` contains images, not test code. |
| Database/config | None | No SQLite database, migrations, environment schema, logging configuration, or recipe storage exists. |

## Current backend behavior

`backend/main.py` and `backend/api.py` each:

1. Create a FastAPI application with unrestricted CORS.
2. Load `./saved_model/best_model.pth` during import.
3. Accept one multipart field named `file` at `POST /predict`.
4. Save it under `uploads/` with a generated `.jpg` filename.
5. Run the PyTorch classifier.
6. write an annotated image to `annotated_output/`.
7. Return only the image file.

Important behavior and risks:

- Paths depend on launching Uvicorn from `backend/`.
- Startup fails when the model or dependencies are unavailable.
- The raw upload is renamed as `.jpg` regardless of its actual format; this is unsuitable for preserving native 16-bit/32-bit depth data.
- There is no file type, file size, decode, image dimension, bit-depth, or pair validation.
- Invalid image decoding is not handled before accessing the image.
- Runtime data is never cleaned up or associated with an inspection record.
- Neuron tensors use shared filenames and are overwritten on every request, creating concurrency and storage risks.
- The API returns no structured decision, confidence, inspection ID, model version, recipe version, latency, or error contract.
- `allow_origins=["*"]` with credentials is too permissive for an industrial deployment.

## Model and training assessment

The current CNN is suitable only as prototype code. Its two convolution layers feed a large fixed-size fully connected layer, tying it to a 224 x 224 preprocessing shape.

Critical correctness issue: `ImageFolder` assigns classes alphabetically as:

1. `dispense_error`
2. `misalignment`
3. `missing_part`
4. `no_defect`

`backend/predictor.py` interprets output indices as:

1. `missing_part`
2. `dispense_error`
3. `misalignment`
4. `no_defect`

Therefore, three of four output indices are interpreted as the wrong label unless the checkpoint was trained by some undocumented alternative mapping. The checkpoint contains no adjacent class-map or training manifest to resolve this uncertainty.

Additional limitations:

- Training uses every image and reports only the final batch loss.
- There is no validation/test split, augmentation policy, normalization, early stopping, reproducibility seed, or evaluation report.
- OpenCV loads color images as BGR while `ToPILImage` treats the array as RGB, so future color inputs would have swapped channels.
- The inference model is duplicated in `model.py` and `predictor.py`, with different forward return values.
- There is no 3D branch, multimodal fusion, uncertainty calibration, ONNX export, or parity test.
- No accuracy claim can be supported by the repository evidence.

## Dataset assessment

The current extracted training dataset contains 32 images:

| Class | Images |
| --- | ---: |
| `dispense_error` | 9 |
| `misalignment` | 1 |
| `missing_part` | 4 |
| `no_defect` | 18 |

Observed image properties:

- Formats: 20 BMP and 12 JPEG.
- Mode: all 32 are grayscale (`L`).
- Sizes: 14 at 1440 x 1080 and 18 at 1920 x 1200.
- 3D/depth candidates: none (`TIFF`, NumPy arrays, point clouds, meshes, or equivalent native-depth files were not found).
- All 32 files in `backend/dataset` are byte-for-byte duplicated in `backend/dataset_raw`.
- All 10 files in `backend/test` are byte-for-byte present in `backend/dataset`, so the folder cannot provide an independent evaluation.
- The class distribution is severely imbalanced; `misalignment` has only one sample.
- Some filenames containing `FAIL` are placed under `no_defect`/`OK`, which requires confirmation from the vision team.

Two additional dataset ZIP files are untracked. They were deliberately not added to Git during this assessment.

## Frontend assessment

The frontend is a default Vite/React JavaScript project with a single upload workflow. It posts to `http://127.0.0.1:8000/predict` and expects an image blob.

Gaps and risks:

- Only one image can be selected; there is no strict 2D/3D pair.
- API URL is hard-coded instead of environment-configured.
- No board, recipe, lot, operator, metadata, history, or engineer view exists.
- Errors are generic and structured backend validation cannot be displayed.
- Created object URLs are not revoked.
- JSX uses Tailwind-style class names, but Tailwind is not installed or configured, and `App.css` is not imported. The intended layout therefore is not implemented by the declared stack.
- No component tests, end-to-end tests, or API mocks exist.

## Git and storage assessment

The repository tracks 158 files, including:

- 32 upload files
- 21 annotated outputs
- 32 processed dataset images
- 32 raw dataset images
- 3 neuron activation files
- 1 saved model

There is no root `.gitignore`. The frontend-only ignore file covers Node artifacts but not Python caches, virtual environments, runtime storage, databases, model outputs, local configuration, or dataset archives.

Existing tracked data must not be removed as part of assessment. A later cleanup requires an explicit, reviewed migration because removing files from Git changes project history and team workflows.

## Verification performed

| Check | Result |
| --- | --- |
| Repository/source inspection | Completed. |
| Git status/history inspection | Completed; tracked code was clean at `b720c31` before adding planning documents. Two dataset ZIPs were already untracked. |
| Dataset count, format, mode, size, and 3D-candidate audit | Completed read-only. |
| Training/test hash comparison | Completed; 10/10 test images duplicate training images. |
| Frontend `npm run build` | Not executed successfully: `node_modules` is absent and Vite is unavailable. Packages were not installed because Task 01 forbids installation without approval. |
| Frontend `npm run lint` | Not executed successfully: `node_modules` is absent and ESLint is unavailable. |
| Backend import smoke test | Failed before application import: active Python is 3.10.0 and FastAPI is not installed. Packages were not installed. |
| Backend automated tests | None configured. |

## Unknowns requiring later confirmation

- Exact 2D camera format, color space, calibration, and acquisition geometry.
- Exact 3D source format, units, invalid-value representation, bit depth, scale, and alignment to 2D.
- Whether inspections are board-level, component-level, or both.
- Required cycle time and acceptable false-pass/false-fail limits by defect type.
- Ground-truth authority and review/correction process.
- Whether the current checkpoint and labels have any trusted provenance.
- Runtime deployment target, GPU availability, authentication, retention, backup, and network constraints.

## Conclusion

The repository provides useful prototype evidence, but not a trustworthy production baseline. Phase 1 should establish a deterministic application foundation and health check while preserving the current `/predict` workflow unchanged. Dataset contracts, paired validation, database storage, mock inference, and UI expansion belong to later phases.
