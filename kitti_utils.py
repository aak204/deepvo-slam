#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Утилиты для работы с KITTI GPS/IMU траекториями
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
import os


def compare_trajectories(gps_poses_file: str, original_poses_file: str, 
                        sequence_id: str, output_dir: str = '.') -> None:
    """
    Сравнивает GPS/IMU траектории с оригинальными poses файлами KITTI.

    Args:
        gps_poses_file: Путь к файлу с GPS/IMU траекториями
        original_poses_file: Путь к оригинальному файлу poses
        sequence_id: Идентификатор последовательности
        output_dir: Директория для сохранения сравнения
    """
    # Загружаем траектории
    gps_poses = np.loadtxt(gps_poses_file)
    original_poses = np.loadtxt(original_poses_file)

    # Извлекаем x, z координаты
    gps_x, gps_z = gps_poses[:, 3], gps_poses[:, 11]
    orig_x, orig_z = original_poses[:, 3], original_poses[:, 11]

    # Создаем сравнительный график
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(gps_x, gps_z, 'b-', label='GPS/IMU', linewidth=2)
    plt.plot(orig_x, orig_z, 'r--', label='Оригинал KITTI', linewidth=2)
    plt.xlabel('X (м)')
    plt.ylabel('Z (м)')
    plt.title(f'Сравнение траекторий {sequence_id}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')

    plt.subplot(2, 2, 2)
    plt.plot(gps_x, 'b-', label='GPS/IMU X', linewidth=2)
    plt.plot(orig_x, 'r--', label='Оригинал X', linewidth=2)
    plt.xlabel('Кадр')
    plt.ylabel('X координата (м)')
    plt.title('X координата по времени')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 3)
    plt.plot(gps_z, 'b-', label='GPS/IMU Z', linewidth=2)
    plt.plot(orig_z, 'r--', label='Оригинал Z', linewidth=2)
    plt.xlabel('Кадр')
    plt.ylabel('Z координата (м)')
    plt.title('Z координата по времени')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 4)
    diff_x = gps_x - orig_x
    diff_z = gps_z - orig_z
    plt.plot(diff_x, 'g-', label='Разность X', linewidth=2)
    plt.plot(diff_z, 'm-', label='Разность Z', linewidth=2)
    plt.xlabel('Кадр')
    plt.ylabel('Разность координат (м)')
    plt.title('Разность между траекториями')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    # Сохраняем график
    output_file = os.path.join(output_dir, f'comparison_{sequence_id}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    # Выводим статистику
    print(f"Статистика для последовательности {sequence_id}:")
    print(f"  Среднее отклонение X: {np.mean(np.abs(diff_x)):.3f} м")
    print(f"  Среднее отклонение Z: {np.mean(np.abs(diff_z)):.3f} м")
    print(f"  Максимальное отклонение X: {np.max(np.abs(diff_x)):.3f} м")
    print(f"  Максимальное отклонение Z: {np.max(np.abs(diff_z)):.3f} м")
    print(f"  Сохранено сравнение: {output_file}")


def convert_poses_to_tum_format(poses_file: str, output_file: str, 
                               timestamps_file: Optional[str] = None) -> None:
    """
    Конвертирует файл poses в формат TUM для использования с evo.

    Args:
        poses_file: Путь к файлу с poses
        output_file: Путь к выходному файлу в формате TUM
        timestamps_file: Путь к файлу с временными метками (опционально)
    """
    poses = np.loadtxt(poses_file)

    # Если временные метки не предоставлены, используем индексы кадров
    if timestamps_file is None:
        timestamps = np.arange(len(poses), dtype=float)
    else:
        timestamps = np.loadtxt(timestamps_file)

    with open(output_file, 'w') as f:
        f.write("# timestamp tx ty tz qx qy qz qw\n")

        for i, pose in enumerate(poses):
            # Восстанавливаем матрицу 3x4
            T = pose.reshape(3, 4)

            # Извлекаем перемещение
            tx, ty, tz = T[0, 3], T[1, 3], T[2, 3]

            # Извлекаем матрицу поворота
            R = T[:3, :3]

            # Конвертируем в кватернион
            qw, qx, qy, qz = rotation_matrix_to_quaternion(R)

            # Записываем в формате TUM
            f.write(f"{timestamps[i]:.6f} {tx:.6f} {ty:.6f} {tz:.6f} "
                   f"{qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n")


def rotation_matrix_to_quaternion(R: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Конвертирует матрицу поворота в кватернион.

    Args:
        R: Матрица поворота 3x3

    Returns:
        Кватернион (w, x, y, z)
    """
    trace = np.trace(R)

    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2  # s = 4 * qw
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2  # s = 4 * qx
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2  # s = 4 * qy
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2  # s = 4 * qz
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s

    return qw, qx, qy, qz


def analyze_trajectory_statistics(poses_file: str, sequence_id: str) -> None:
    """
    Анализирует статистику траектории.

    Args:
        poses_file: Путь к файлу с poses
        sequence_id: Идентификатор последовательности
    """
    poses = np.loadtxt(poses_file)

    # Извлекаем координаты
    x_coords = poses[:, 3]
    y_coords = poses[:, 7]
    z_coords = poses[:, 11]

    # Вычисляем расстояния между соседними позами
    distances = []
    for i in range(1, len(poses)):
        dx = x_coords[i] - x_coords[i-1]
        dy = y_coords[i] - y_coords[i-1]
        dz = z_coords[i] - z_coords[i-1]
        dist = np.sqrt(dx**2 + dy**2 + dz**2)
        distances.append(dist)

    distances = np.array(distances)

    # Вычисляем общую длину пути
    total_distance = np.sum(distances)

    # Вычисляем размеры области
    x_range = np.max(x_coords) - np.min(x_coords)
    y_range = np.max(y_coords) - np.min(y_coords)
    z_range = np.max(z_coords) - np.min(z_coords)

    print(f"\nСтатистика траектории {sequence_id}:")
    print(f"  Количество кадров: {len(poses)}")
    print(f"  Общая длина пути: {total_distance:.2f} м")
    print(f"  Средняя скорость: {np.mean(distances):.3f} м/кадр")
    print(f"  Максимальная скорость: {np.max(distances):.3f} м/кадр")
    print(f"  Размер области X: {x_range:.2f} м")
    print(f"  Размер области Y: {y_range:.2f} м") 
    print(f"  Размер области Z: {z_range:.2f} м")
    print(f"  Начальная позиция: ({x_coords[0]:.2f}, {y_coords[0]:.2f}, {z_coords[0]:.2f})")
    print(f"  Конечная позиция: ({x_coords[-1]:.2f}, {y_coords[-1]:.2f}, {z_coords[-1]:.2f})")


def main():
    """
    Пример использования утилит.
    """
    import argparse

    parser = argparse.ArgumentParser(description='Утилиты для KITTI GPS/IMU траекторий')
    parser.add_argument('--analyze', type=str, help='Анализировать траекторию')
    parser.add_argument('--compare', nargs=2, help='Сравнить две траектории')
    parser.add_argument('--convert_tum', nargs=2, help='Конвертировать в формат TUM')
    parser.add_argument('--sequence', type=str, default='00', help='Идентификатор последовательности')

    args = parser.parse_args()

    if args.analyze:
        analyze_trajectory_statistics(args.analyze, args.sequence)

    if args.compare:
        compare_trajectories(args.compare[0], args.compare[1], args.sequence)

    if args.convert_tum:
        convert_poses_to_tum_format(args.convert_tum[0], args.convert_tum[1])
        print(f"Конвертировано в формат TUM: {args.convert_tum[1]}")


if __name__ == '__main__':
    main()
