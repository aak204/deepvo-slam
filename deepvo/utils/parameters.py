"""
Параметры конфигурации для DeepVO

Содержит все настройки модели, обучения, путей к данным и экспериментов.
"""

import os
import torch


class Parameters:
    """Класс параметров конфигурации для системы DeepVO."""
    
    def __init__(self, fine_tuning_mode=False):
        # Системные параметры
        self.n_processors = 12
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Пути к данным
        self.data_dir = './dataset'
        self.image_dir = os.path.join(self.data_dir, 'sequences')
        self.pose_dir = os.path.join(self.data_dir, 'poses')
        
        # Списки последовательностей для обучения, валидации и теста
        self.train_video = ['00', '02', '08', '09']
        self.valid_video = ['03', '05']
        self.test_video = ['03', '04', '05', '06', '07', '10']

        # Параметры последовательности
        self.seq_len = 7
        self.overlap = 1

        # Параметры изображений
        self.img_w = 640
        self.img_h = 192

        # Параметры аугментации данных
        self.is_hflip = False
        self.is_color_jitter = False
        self.brightness = 0.4
        self.contrast = 0.4
        self.saturation = 0.4
        self.hue = 0.1
        
        # Настройки нейронной сети
        self.rnn_hidden_size = 1000
        self.conv_dropout = (0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.5)
        self.rnn_dropout_out = 0.2
        self.rnn_dropout_between = 0.2
        self.batch_norm = True

        # Настройки обучения
        if fine_tuning_mode:
            self.epochs = 20
            self.optim_lr = 5e-5
            self.experiment_name = 'kitti_finetuned'
        else:
            self.epochs = 130
            self.optim_lr = 1e-3
            self.experiment_name = 'kitti_original_params'
            
        self.batch_size = 12
        self.pin_mem = True
        self.optim_decay = 5e-6
        self.optim_lr_decay_factor = 0.1
        self.optim_lr_step = 60

        # Пути для предобученной модели
        self.pretrained_flownet = './models/flownets_bn_EPE2.459.pth'
        self.resume = fine_tuning_mode
        self.resume_t_or_v = '.latest'
        
        # Пути для сохранения и загрузки модели
        self.save_path = f'experiments/{self.experiment_name}'

        self.name = (
            f't{"".join(self.train_video)}_v{"".join(self.test_video)}_'
            f'im{self.img_h}x{self.img_w}_s{self.seq_len}_b{self.batch_size}'
        )
        if self.is_hflip:
            self.name += '_flip'
        if fine_tuning_mode:
            self.name += '_finetuned'

        self.load_model_path = f'{self.save_path}/models/{self.name}.model{self.resume_t_or_v}'
        self.load_optimizer_path = f'{self.save_path}/models/{self.name}.optimizer{self.resume_t_or_v}'
        self.record_path = f'{self.save_path}/records/{self.name}.txt'
        self.save_model_path = f'{self.save_path}/models/{self.name}.model'
        self.save_optimzer_path = f'{self.save_path}/models/{self.name}.optimizer'

        # Автоматическое создание папок
        os.makedirs(os.path.dirname(self.record_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.save_model_path), exist_ok=True)


# Создаем экземпляр параметров по умолчанию
par = Parameters()