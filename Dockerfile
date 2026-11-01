FROM python:3.11-slim
ARG WITH_ML=0
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt requirements-ml.txt ./
RUN pip install --no-cache-dir -r requirements.txt && if [ "$WITH_ML" = "1" ]; then pip install --no-cache-dir -r requirements-ml.txt; fi
COPY . .
EXPOSE 8000
CMD ["uvicorn", "src.serve:app", "--host", "0.0.0.0", "--port", "8000"]
