FROM python:3.12-slim

WORKDIR /app

COPY index.html /app/index.html

EXPOSE 8000

CMD ["python3", "-m", "http.server", "8000", "--directory", "/app"]
