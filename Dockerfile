FROM python:3.11-slim

WORKDIR /app
COPY permissions.json .
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY signal_bot.py .

CMD ["python", "signal_bot.py"]
