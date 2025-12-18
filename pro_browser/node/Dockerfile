FROM mcr.microsoft.com/playwright/python:v1.57.0-jammy

WORKDIR /app

# Install system dependencies for aiortc/av/opencv
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    libavdevice-dev \
    libavfilter-dev \
    libopus-dev \
    libvpx-dev \
    pkg-config \
    libsrtp2-dev \
    libnss-mdns \
    avahi-daemon \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

# Environment variables for production
ENV PYTHONUNBUFFERED=1

CMD ["python", "main.py"]
