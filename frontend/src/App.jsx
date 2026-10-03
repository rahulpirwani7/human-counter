import { useState } from "react";
import "./App.css";

function App() {
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [count, setCount] = useState(null);
  const [resultImage, setResultImage] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleImageChange = (event) => {
    const file = event.target.files[0];

    if (file) {
      setImage(file);
      setPreview(URL.createObjectURL(file));
      setCount(null);
      setResultImage(null);
      setError("");
    }
  };

  const handleDetect = async () => {
    if (!image) return;

    setLoading(true);
    setError("");

    const formData = new FormData();
    formData.append("image", image);

    try {
      const response = await fetch(
        "/detect_yolo",
        {
          method: "POST",
          body: formData,
        }
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || "Detection failed");
      }

      setCount(data.count);
      setResultImage(data.image);
    } catch (error) {
      setError(error.message);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setImage(null);
    setPreview(null);
    setCount(null);
    setResultImage(null);
    setError("");
  };

  return (
    <div className="app-container">
      <div className="main-card">
        <div className="header">
          <div className="logo">HC</div>

          <div className="status">
            <span></span>
            Model Ready
          </div>
        </div>

        <div className="hero">
          <h1>Human Counter</h1>

          <p>
            Upload an image to detect and count people using YOLO.
          </p>
        </div>

        <div className="content-grid">
          <div className="section">
            <div className="section-header">
              <div>
                <span className="section-number">01</span>
                <h2>Upload Image</h2>
              </div>

              {image && (
                <button
                  className="reset-button"
                  onClick={handleReset}
                >
                  Reset
                </button>
              )}
            </div>

            {!preview ? (
              <label className="upload-box">
                <div className="upload-icon">↑</div>

                <h3>Choose an image</h3>

                <p>JPG, JPEG or PNG</p>

                <span className="browse-button">
                  Browse Image
                </span>

                <input
                  type="file"
                  accept="image/*"
                  onChange={handleImageChange}
                  hidden
                />
              </label>
            ) : (
              <div className="preview-box">
                <img
                  src={preview}
                  alt="Selected"
                  className="preview-image"
                />

                <div className="file-info">
                  <span>{image.name}</span>

                  <label className="change-button">
                    Change
                    <input
                      type="file"
                      accept="image/*"
                      onChange={handleImageChange}
                      hidden
                    />
                  </label>
                </div>
              </div>
            )}

            <button
              className="detect-button"
              onClick={handleDetect}
              disabled={!image || loading}
            >
              {loading ? "Detecting..." : "Detect People"}
            </button>

            {error && (
              <div className="error-message">
                {error}
              </div>
            )}
          </div>

          <div className="section">
            <div className="section-header">
              <div>
                <span className="section-number">02</span>
                <h2>Detection Result</h2>
              </div>
            </div>

            {!resultImage ? (
              <div className="empty-result">
                <div className="result-icon">◎</div>

                <h3>No detection yet</h3>

                <p>
                  Your detection result will appear here.
                </p>
              </div>
            ) : (
              <div className="result-content">
                <div className="count-box">
                  <span>People Detected</span>
                  <strong>{count}</strong>
                </div>

                <img
                  src={resultImage}
                  alt="Detection Result"
                  className="result-image"
                />
              </div>
            )}
          </div>
        </div>

        <div className="footer">
          TensorFlow • YOLO • Flask • React
        </div>
      </div>
    </div>
  );
}

export default App;