#!/usr/bin/env python3
"""
Базовый пример использования DeepVO SLAM системы

Показывает как:
1. Загрузить конфигурацию
2. Инициализировать SLAM систему
3. Запустить обработку последовательности
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deepvo.slam import SLAMSystem
from deepvo.utils import Parameters


def main():
    """Основная функция примера."""
    print("=== Пример использования DeepVO SLAM ===")
    
    # Настройка параметров
    config = Parameters()
    
    # Инициализация SLAM системы
    slam_system = SLAMSystem(config)
    
    # Настройка путей (измените на ваши)
    sequence_id = '04'
    kitti_raw_dir = './kitti_raw_data'
    image_dir = './dataset/sequences'
    model_path = './experiments/kitti_finetuned/models/model.latest'
    graphml_file = f'./map_graphs/map_graph_{sequence_id}.graphml'
    result_dir = './results'
    
    try:
        # Загрузка модели
        slam_system.load_model(model_path)
        
        # Запуск SLAM
        trajectory = slam_system.run_slam(
            sequence_id=sequence_id,
            kitti_raw_dir=kitti_raw_dir,
            image_dir=image_dir,
            graphml_file=graphml_file,
            result_dir=result_dir
        )
        
        print(f"✅ SLAM завершен успешно! Получена траектория из {len(trajectory)} поз.")
        
    except FileNotFoundError as e:
        print(f"❌ Ошибка: {e}")
        print("Убедитесь, что все необходимые файлы находятся в правильных директориях.")
    except Exception as e:
        print(f"❌ Произошла ошибка: {e}")


if __name__ == '__main__':
    main()