# 1. Base Image: Lightweight Debian Python
FROM mirror.gcr.io/library/python:3.10-slim

# 2. System Variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEFECT_AREA_THRESHOLD=50

# 3. Working Directory
WORKDIR /workspace

# 4. Layer Caching: Install dependencies first
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Inject Microservice Code
COPY app/ app/
COPY models/ models/

# 6. Expose the API Port
EXPOSE 8000

# 7. Execute the Engine
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]