FROM python:3.12-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MUJOCO_GL=osmesa OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
RUN apt-get update && apt-get install -y --no-install-recommends libosmesa6 libgl1 libglfw3 ffmpeg build-essential linux-libc-dev && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace
COPY requirements.txt requirements-lock.txt ./
RUN pip install --no-cache-dir torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu && pip install --no-cache-dir -r requirements-lock.txt
COPY . .
CMD ["bash"]
