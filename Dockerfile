FROM python:3.10-slim

# Устанавливаем рабочую директорию внутри контейнера
WORKDIR /app

# Устанавливаем переменные окружения
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED 1

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libboost-all-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/borglab/gtsam.git && \
    cd gtsam && \
    mkdir build && \
    cd build && \
    cmake .. -DGTSAM_BUILD_PYTHON=ON -DGTSAM_WITH_TBB=OFF -DPYTHON_EXECUTABLE=$(which python) && \
    # Компилируем зависимости и саму библиотеку
    make -j$(nproc) && \
    make python-install && \
    # Очищаем исходники для уменьшения размера образа
    cd / && rm -rf /gtsam

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Этап 6: Настройка запуска
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
