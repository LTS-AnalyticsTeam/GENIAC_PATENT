FROM python:3.11-slim

WORKDIR /app

RUN pip install fastapi==0.111.0 uvicorn[standard]==0.30.1

COPY app.py /app/app.py

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "9000"]

