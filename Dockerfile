FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
# GUI deps (PySide6 etc.) are not needed in the container
RUN grep -vE '^(PySide6|pyqtgraph|sounddevice)' requirements.txt \
        > req-web.txt && pip install --no-cache-dir -r req-web.txt
COPY rvsangeetha ./rvsangeetha
EXPOSE 8080
CMD ["uvicorn", "rvsangeetha.web.server:app", "--host", "0.0.0.0", "--port", "8080"]
