FROM python:3.9

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libssl-dev \
        libffi-dev \
        libpq-dev \
        libgl1-mesa-glx \
        && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/
RUN pip install -r requirements.txt

COPY . /app/

CMD ["uvicorn", "app.app:app", "--reload", "--workers", "1", "--host", "0.0.0.0", "--port", "8000"]