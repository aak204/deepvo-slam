"""
Базовые тесты для DeepVO системы
"""

import pytest
import torch
import numpy as np
import sys
import os

# Добавляем корневую директорию в path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deepvo.slam.deepvo_model import DeepVO
from deepvo.utils.parameters import Parameters


class TestDeepVO:
    """Тесты для модели DeepVO."""
    
    def setup_method(self):
        """Настройка для каждого теста."""
        self.params = Parameters()
        self.model = DeepVO(
            self.params.img_h, 
            self.params.img_w, 
            self.params.batch_norm
        )
    
    def test_model_initialization(self):
        """Тест инициализации модели."""
        assert self.model is not None
        assert hasattr(self.model, 'conv1')
        assert hasattr(self.model, 'rnn')
        assert hasattr(self.model, 'linear')
    
    def test_model_forward_pass(self):
        """Тест прямого прохода модели."""
        batch_size = 2
        seq_len = 3
        channels = 3
        height = self.params.img_h
        width = self.params.img_w
        
        # Создаем тестовый тензор
        x = torch.randn(batch_size, seq_len, channels, height, width)
        
        # Прямой проход
        angle, trans, hc = self.model(x)
        
        # Проверяем размеры выходов
        expected_seq_len = seq_len - 1  # Из-за попарной обработки изображений
        assert angle.shape == (batch_size, expected_seq_len, 3)
        assert trans.shape == (batch_size, expected_seq_len, 3)
        assert hc is not None
    
    def test_model_loss_computation(self):
        """Тест вычисления функции потерь."""
        batch_size = 2
        seq_len = 3
        channels = 3
        height = self.params.img_h
        width = self.params.img_w
        
        x = torch.randn(batch_size, seq_len, channels, height, width)
        y = torch.randn(batch_size, seq_len - 1, 6)  # 6DoF ground truth
        
        loss, angle_loss, trans_loss = self.model.get_loss(x, y)
        
        assert isinstance(loss.item(), float)
        assert isinstance(angle_loss.item(), float)
        assert isinstance(trans_loss.item(), float)
        assert loss.item() >= 0
    
    def test_model_encode_image(self):
        """Тест кодирования изображений."""
        batch_size = 2
        channels = 6  # Стековые изображения
        height = self.params.img_h
        width = self.params.img_w
        
        x = torch.randn(batch_size, channels, height, width)
        encoded = self.model.encode_image(x)
        
        assert encoded is not None
        assert len(encoded.shape) == 4  # (batch, channels, height, width)


class TestParameters:
    """Тесты для класса Parameters."""
    
    def test_parameters_initialization(self):
        """Тест инициализации параметров."""
        params = Parameters()
        
        assert params.img_h == 192
        assert params.img_w == 640
        assert params.seq_len == 7
        assert params.batch_size == 12
        assert isinstance(params.train_video, list)
        assert isinstance(params.valid_video, list)
    
    def test_fine_tuning_mode(self):
        """Тест режима fine-tuning."""
        params_normal = Parameters(fine_tuning_mode=False)
        params_finetune = Parameters(fine_tuning_mode=True)
        
        assert params_normal.epochs == 130
        assert params_finetune.epochs == 20
        assert params_normal.optim_lr == 1e-3
        assert params_finetune.optim_lr == 5e-5


if __name__ == '__main__':
    pytest.main([__file__])