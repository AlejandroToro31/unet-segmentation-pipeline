# U-Net Semantic Segmentation API: Precision Defect Localization

![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker)

An enterprise-grade, containerized Machine Learning microservice engineered for high-precision, pixel-level defect localization on industrial manufacturing lines. 

Built with strict software engineering standards, this API utilizes a U-Net architecture to semantically segment anomalous elements. Instead of simply classifying an image as defective, the system mathematically maps the exact boundaries of the anomaly and transmits a compressed Base64 visual mask directly back to the client dashboard for zero-latency overlay rendering.

## System Architecture

* **The Engine:** PyTorch `segmentation_models_pytorch` utilizing a U-Net with a pre-trained ResNet34 encoder.
* **The Optimization:** Trained using a custom Compound Loss function (Binary Cross Entropy + Dice Loss) to combat the severe class imbalance inherent in anomaly detection.
* **The Preprocessor:** High-speed `cv2.imdecode` combined with `Albumentations` for direct-memory byte-stream transformations, eliminating disk I/O latency.
* **The Server:** Asynchronous FastAPI providing a strict Pydantic-validated REST endpoint.
* **The Payload:** Intelligent Base64 encoding. The API only transmits heavy mask image data over the network if a defect actually breaches the threshold, optimizing bandwidth.

## Quick Start (Production Environment)

The microservice is fully isolated via Docker. No local Python environment is required.

### 0. Download the Model Artifact
Due to GitHub file constraints, the trained U-Net weights are hosted externally.
1. Download `best_unet_bottle.pth` from this direct link: **[https://drive.google.com/file/d/1jRp_T8NHWgksMLqPqvSAfUuQhHjZYw7l/view?usp=sharing]**
2. Place the downloaded `.pth` file directly inside the `models/` directory of this repository.

### 1. Build the Microservice
```bash
docker build -t unet-api:latest .
```
### 2. Ignite the Container
By default, the API will flag a defect if more than 50 anomalous pixels are detected. You can override this dynamically at runtime without rebuilding the container:

```Bash
docker run -p 8000:8000 -e DEFECT_AREA_THRESHOLD=50 unet-api:latest
```
### 3. Execute Inference
Navigate to http://127.0.0.1:8000/docs to access the interactive Swagger UI.
Upload an image to the /segment/ endpoint. The API will evaluate the pixel array and return a real-time JSON payload containing the defect count, the boolean classification, and the Base64-encoded visual mask string for frontend UI injection.