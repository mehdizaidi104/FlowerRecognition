# This is the "recipe" for the trainer container
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY train.py .
CMD ["python", "train.py"]