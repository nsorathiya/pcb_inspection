import React, { useState } from "react";
import axios from "axios";

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [status, setStatus] = useState("");

  const handleUpload = async () => {
    if (!file) {
      setStatus("⚠️ Please select a file before uploading.");
      return;
    }

    setStatus("Uploading...");

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await axios.post("http://127.0.0.1:8000/predict", formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
        responseType: "blob",
      });

      const contentType = response.headers["content-type"];
      console.log("Received content-type:", contentType);

      if (contentType?.startsWith("image/")) {
        const imageURL = URL.createObjectURL(response.data);
        setPreview(imageURL);
        setStatus("✅ Upload successful! Prediction complete.");
      } else {
        const reader = new FileReader();
        reader.onload = () => {
          console.log("Non-image response:", reader.result);
          setStatus("❌ Unexpected server response.");
        };
        reader.readAsText(response.data);
      }
    } catch (error) {
      console.error("Upload error:", error);
      setStatus("❌ Upload failed. Please check backend and try again.");
    }
  };

  return (
    <div className="min-h-screen bg-gray-100 p-8">
      <div className="max-w-md mx-auto bg-white p-6 rounded-xl shadow">
        <h1 className="text-xl font-bold mb-4">PCB Defect Inspector</h1>

        <input
          type="file"
          accept="image/*"
          onChange={(e) => {
            setFile(e.target.files[0]);
            setStatus("");
            setPreview(null);
          }}
          className="mb-4"
        />

        <button
          onClick={handleUpload}
          className="px-4 py-2 bg-blue-600 text-white rounded"
        >
          Upload and Inspect
        </button>

        {status && (
          <div className="mt-4 text-sm text-gray-800 font-medium">
            {status}
          </div>
        )}

        {preview && (
          <div className="mt-4">
            <h2 className="font-semibold">Prediction Result:</h2>
            <img src={preview} alt="Prediction Result" className="rounded-xl mt-2" />
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
