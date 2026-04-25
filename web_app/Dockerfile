FROM python:3.11-slim

WORKDIR /app

# Runtime libs used by torch/scipy wheels
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

COPY requirements_frontend.txt .

RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir --only-binary=:all: -r requirements_frontend.txt

COPY . .

ENV PYTHONUNBUFFERED=1
ENV PORT=10000

CMD ["sh", "-c", "gunicorn app:app --bind 0.0.0.0:${PORT} --workers 1 --threads 8 --timeout 300"]
