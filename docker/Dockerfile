FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

WORKDIR /workspace/MedStyleAudit
COPY pyproject.toml requirements.txt README.md ./
COPY src ./src
RUN python -m pip install --no-cache-dir -r requirements.txt && python -m pip install --no-cache-dir -e .
COPY . .
ENV PYTHONUNBUFFERED=1
CMD ["python", "-m", "pytest", "-q"]
