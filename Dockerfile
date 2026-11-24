FROM python@sha256:4840d514c583a4d3d8cad3940e68dee6f0f6f3635c12fcd0a3d74c21a5b93a54

WORKDIR /app
COPY requirements-models.txt .
RUN pip install --no-cache-dir -r requirements-models.txt

COPY data/who/codes.jsonl.gz data/who/codes.jsonl.gz
COPY data/who/NOTICE.md data/who/NOTICE.md
COPY data/shorthand.json data/shorthand.json
COPY coder coder
COPY examples/notes.jsonl examples/notes.jsonl

ENV PYTHONUNBUFFERED=1
ENTRYPOINT ["python", "-m", "coder"]
