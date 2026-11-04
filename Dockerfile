FROM python:3.11-slim
ARG WITH_ML=0
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt requirements-ml.txt ./
RUN pip install --no-cache-dir -r requirements.txt && if [ "$WITH_ML" = "1" ]; then pip install --no-cache-dir -r requirements-ml.txt; fi
# mediapipe pulls in opencv-contrib-python, which overlaps opencv-python-headless and can leave cv2 broken
# (e.g. "module 'cv2' has no attribute 'CascadeClassifier'"). Keep exactly one OpenCV build.
RUN pip uninstall -y opencv-python opencv-contrib-python opencv-contrib-python-headless opencv-python-headless \
    && pip install --no-cache-dir "opencv-python-headless>=4.9,<5" \
    && python -c "import cv2; assert hasattr(cv2, 'CascadeClassifier'), cv2.__version__; print('opencv', cv2.__version__)"
COPY . .
EXPOSE 8000
CMD ["uvicorn", "src.serve:app", "--host", "0.0.0.0", "--port", "8000"]
