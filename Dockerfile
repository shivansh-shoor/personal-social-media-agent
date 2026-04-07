FROM python:3.10-slim

RUN useradd -m -u 1000 appuser
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s CMD curl -f http://localhost:7860/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
