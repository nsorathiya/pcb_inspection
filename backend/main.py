from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil, os, uuid, cv2
from predictor import Predictor

# ✅ Step 1: Create app
app = FastAPI()

# ✅ Step 2: Apply CORS here
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Use ["http://localhost:5173"] in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Step 3: Model
predictor = Predictor(model_path="./saved_model/best_model.pth")

# ✅ Step 4: Predict endpoint
@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    temp_filename = f"temp_{uuid.uuid4().hex}.jpg"
    file_path = os.path.join("uploads", temp_filename)
    os.makedirs("uploads", exist_ok=True)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    label, image = predictor.predict_image(file_path)
    result_path = os.path.join("annotated_output", f"pred_{label}_{temp_filename}")
    os.makedirs("annotated_output", exist_ok=True)
    cv2.putText(image, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.rectangle(image, (5, 5), (image.shape[1] - 5, image.shape[0] - 5), (0, 0, 255), 2)
    cv2.imwrite(result_path, image)

    return FileResponse(result_path, media_type="image/jpeg")
