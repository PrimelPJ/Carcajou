FROM python:3.12-slim

LABEL org.opencontainers.image.title="carcajou" \
      org.opencontainers.image.description="GNSS-denied vehicle navigation benchmark" \
      org.opencontainers.image.source="https://github.com/PrimelPJ/Carcajou" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.authors="Primel Jayawardana"

WORKDIR /app

# Install dependencies first so this layer is cached on code-only changes.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir -e ".[plots]"

COPY scripts ./scripts
COPY tests ./tests

ENTRYPOINT ["python", "scripts/run_benchmark.py"]
CMD ["--laps", "5", "--seeds", "3"]
