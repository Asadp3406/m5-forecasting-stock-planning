# Build from the project root:  docker build -f api/Dockerfile -t m5-api .
FROM python:3.12-slim

WORKDIR /code
COPY api/requirements.txt api/requirements.txt
RUN pip install --no-cache-dir -r api/requirements.txt

COPY api/main.py api/main.py

# Database settings are NOT stored in the image. Pass them at run time:
#   docker run -p 8000:8000 --env-file .env m5-api
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
