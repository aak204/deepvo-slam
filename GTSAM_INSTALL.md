# Установка GTSAM

GTSAM (Georgia Tech Smoothing and Mapping) - это библиотека для SLAM, которая не включена в репозиторий из-за большого размера (~3GB).

## Варианты установки

### 1. Установка через conda (рекомендуется)

```bash
conda install -c conda-forge gtsam
```

### 2. Установка через pip

```bash
pip install gtsam
```

### 3. Сборка из исходников

```bash
# Клонируем GTSAM
git clone https://github.com/borglab/gtsam.git
cd gtsam

# Создаем build директорию
mkdir build
cd build

# Конфигурируем CMake
cmake ..

# Собираем
make -j4

# Устанавливаем
make install
```

## Проверка установки

```python
import gtsam
print("GTSAM version:", gtsam.__version__)
```

## Зависимости для сборки

- CMake 3.10+
- Boost 1.65+
- Eigen 3.3+
- Intel MKL (опционально)

## Примечание

Если у вас возникают проблемы с установкой GTSAM, попробуйте использовать Docker образ с предустановленным GTSAM или обратитесь к официальной документации: https://github.com/borglab/gtsam 