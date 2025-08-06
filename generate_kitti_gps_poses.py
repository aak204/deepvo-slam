#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Генерация файлов ground truth траекторий для KITTI Odometry последовательностей 00-10
на основе GPS/IMU данных из raw data с использованием библиотеки pykitti.

Автор: Генерировано автоматически
"""

import os
import argparse
import numpy as np
import pykitti
from tqdm import tqdm
from typing import Dict, Tuple, List, Optional
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Таблица соответствия последовательностей Odometry с Raw Data
SEQUENCE_MAPPING = {
    '00': {'date': '2011_10_03', 'drive': '0027', 'frames': (0, 4540)},
    '01': {'date': '2011_10_03', 'drive': '0042', 'frames': (0, 1100)},
    '02': {'date': '2011_10_03', 'drive': '0034', 'frames': (0, 4660)},
    '03': {'date': '2011_09_26', 'drive': '0067', 'frames': (0, 800)},
    '04': {'date': '2011_09_30', 'drive': '0016', 'frames': (0, 270)},
    '05': {'date': '2011_09_30', 'drive': '0018', 'frames': (0, 2760)},
    '06': {'date': '2011_09_30', 'drive': '0020', 'frames': (0, 1100)},
    '07': {'date': '2011_09_30', 'drive': '0027', 'frames': (0, 1100)},
    '08': {'date': '2011_09_30', 'drive': '0028', 'frames': (1100, 5170)},
    '09': {'date': '2011_09_30', 'drive': '0033', 'frames': (0, 1590)},
    '10': {'date': '2011_09_30', 'drive': '0034', 'frames': (0, 1200)},
}


def load_raw_data(basedir: str, date: str, drive: str, 
                  frame_range: Tuple[int, int]) -> pykitti.raw:
    """
    Загружает raw данные KITTI для указанной поездки и диапазона кадров.

    Args:
        basedir: Базовая директория с raw данными
        date: Дата поездки (например, '2011_10_03')
        drive: Номер поездки (например, '0027')
        frame_range: Диапазон кадров (start, end)

    Returns:
        Объект pykitti.raw с загруженными данными
    """
    start_frame, end_frame = frame_range
    frames = list(range(start_frame, end_frame + 1))

    try:
        dataset = pykitti.raw(basedir, date, drive, frames=frames)
        return dataset
    except Exception as e:
        raise RuntimeError(f"Ошибка загрузки данных {date}/{drive}: {e}")


def extract_poses_from_oxts(dataset: pykitti.raw) -> np.ndarray:
    """
    Извлекает матрицы поз из данных OXTS (GPS/IMU).

    Args:
        dataset: Объект pykitti.raw с загруженными данными

    Returns:
        Массив матриц поз размера (N, 4, 4)
    """
    poses = []

    for oxts in dataset.oxts:
        # Получаем матрицу преобразования T_w_imu (4x4)
        # pykitti автоматически вычисляет эту матрицу из GPS/IMU данных
        T_w_imu = oxts.T_w_imu
        poses.append(T_w_imu)

    return np.array(poses)


def poses_to_kitti_format(poses: np.ndarray) -> np.ndarray:
    """
    Преобразует матрицы поз 4x4 в формат KITTI (3x4 сплющенный).

    Args:
        poses: Массив матриц поз размера (N, 4, 4)

    Returns:
        Массив поз в формате KITTI (N, 12)
    """
    # Берем верхние 3 строки матрицы 4x4 и сплющиваем их
    kitti_poses = poses[:, :3, :].reshape(-1, 12)
    return kitti_poses


def save_trajectory(poses: np.ndarray, output_file: str) -> None:
    """
    Сохраняет траекторию в файл в формате KITTI.

    Args:
        poses: Массив поз в формате KITTI (N, 12)
        output_file: Путь к выходному файлу
    """
    with open(output_file, 'w') as f:
        for pose in poses:
            # Форматируем числа с плавающей запятой с достаточной точностью
            pose_str = ' '.join([f'{x:.6f}' for x in pose])
            f.write(pose_str + '\n')


def visualize_trajectory(poses: np.ndarray, sequence_id: str, 
                        output_dir: str) -> None:
    """
    Визуализирует 2D траекторию (x, z координаты).

    Args:
        poses: Массив поз в формате KITTI (N, 12)
        sequence_id: Идентификатор последовательности
        output_dir: Директория для сохранения графика
    """
    # Извлекаем x, z координаты (столбцы 3 и 11 в формате KITTI)
    x_coords = poses[:, 3]   # T03
    z_coords = poses[:, 11]  # T23

    plt.figure(figsize=(10, 8))
    plt.plot(x_coords, z_coords, 'b-', linewidth=2, label='Траектория')
    plt.plot(x_coords[0], z_coords[0], 'go', markersize=10, label='Старт')
    plt.plot(x_coords[-1], z_coords[-1], 'ro', markersize=10, label='Финиш')

    plt.xlabel('X координата (м)')
    plt.ylabel('Z координата (м)')
    plt.title(f'Траектория последовательности {sequence_id} (GPS/IMU)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')

    # Сохраняем график
    plot_file = os.path.join(output_dir, f'trajectory_{sequence_id}.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()


def process_sequence(sequence_id: str, basedir: str, 
                    output_dir: str, visualize: bool = False) -> None:
    """
    Обрабатывает одну последовательность KITTI.

    Args:
        sequence_id: Идентификатор последовательности ('00', '01', ...)
        basedir: Базовая директория с raw данными
        output_dir: Выходная директория
        visualize: Создавать ли визуализацию траектории
    """
    if sequence_id not in SEQUENCE_MAPPING:
        raise ValueError(f"Неизвестная последовательность: {sequence_id}")

    seq_info = SEQUENCE_MAPPING[sequence_id]
    date = seq_info['date']
    drive = seq_info['drive']
    frame_range = seq_info['frames']

    print(f"Обрабатывается последовательность {sequence_id}: {date}/{drive}, "
          f"кадры {frame_range[0]}-{frame_range[1]}")

    # Загружаем raw данные
    dataset = load_raw_data(basedir, date, drive, frame_range)

    # Извлекаем позы из OXTS данных
    poses_4x4 = extract_poses_from_oxts(dataset)

    # Преобразуем в формат KITTI
    poses_kitti = poses_to_kitti_format(poses_4x4)

    # Сохраняем в файл
    output_file = os.path.join(output_dir, f'{sequence_id}.txt')
    save_trajectory(poses_kitti, output_file)

    print(f"Сохранено {len(poses_kitti)} поз в файл: {output_file}")

    # Создаем визуализацию если требуется
    if visualize:
        visualize_trajectory(poses_kitti, sequence_id, output_dir)
        print(f"Визуализация сохранена: trajectory_{sequence_id}.png")


def main():
    parser = argparse.ArgumentParser(
        description='Генерация GPS/IMU ground truth траекторий для KITTI Odometry',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--input_dir', '-i',
        type=str,
        required=True,
        help='Путь к директории с raw данными KITTI'
    )

    parser.add_argument(
        '--output_dir', '-o',
        type=str,
        default='poses_gps_global',
        help='Выходная директория для сохранения траекторий'
    )

    parser.add_argument(
        '--sequences', '-s',
        type=str,
        nargs='+',
        default=[f'{i:02d}' for i in range(11)],
        help='Список последовательностей для обработки'
    )

    parser.add_argument(
        '--visualize', '-v',
        action='store_true',
        help='Создавать визуализацию траекторий'
    )

    args = parser.parse_args()

    # Проверяем входную директорию
    if not os.path.exists(args.input_dir):
        raise FileNotFoundError(f"Входная директория не найдена: {args.input_dir}")

    # Создаем выходную директорию
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Входная директория: {args.input_dir}")
    print(f"Выходная директория: {args.output_dir}")
    print(f"Последовательности для обработки: {args.sequences}")
    print(f"Визуализация: {'включена' if args.visualize else 'отключена'}")
    print()

    # Обрабатываем каждую последовательность
    for seq_id in tqdm(args.sequences, desc="Обработка последовательностей"):
        try:
            process_sequence(seq_id, args.input_dir, args.output_dir, args.visualize)
        except Exception as e:
            print(f"Ошибка обработки последовательности {seq_id}: {e}")
            continue

    print(f"\nГотово! Файлы траекторий сохранены в: {args.output_dir}")


if __name__ == '__main__':
    main()
