FROM python:3.12-slim

WORKDIR /test

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app.py .

EXPOSE 6325

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "6325"]