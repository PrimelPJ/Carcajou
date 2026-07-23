FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir -e ".[plots]"
COPY scripts ./scripts
COPY tests ./tests

ENTRYPOINT ["python", "scripts/run_benchmark.py"]
CMD ["--laps", "5", "--seeds", "3"]
