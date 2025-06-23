FROM python:3.11-slim

# ❶ パッケージインストール
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ❷ アプリケーションコード
COPY . .

# ❸ Uvicorn で起動
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "3000", "--reload"]
