FROM python:3.11-slim

WORKDIR /app

# ffmpeg/build tools no longer needed — LLM-only path has no faiss/opencv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Gradio listens on 8080 (matches wrangler container defaultPort)
ENV PORT=8080
EXPOSE 8080

CMD ["python", "-m", "interfaces.web_cf"]
