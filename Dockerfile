FROM python:3.11-slim

WORKDIR /app

# System deps for faiss and PIL
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Render uses /data for persistent disk
# First run: copy index + frames to persistent disk
# Subsequent runs: read from persistent disk
ENV GRADIO_SERVER_NAME=0.0.0.0
EXPOSE 10000

CMD ["python", "-m", "interfaces.web_render"]
