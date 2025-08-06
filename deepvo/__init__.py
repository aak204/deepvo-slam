"""
DeepVO: Visual-Inertial SLAM System

Система визуально-инерциальной SLAM на основе глубокого обучения.
"""

__version__ = "1.0.0"
__author__ = "Andrew Korchemkin"

from .slam.deepvo_model import DeepVO
from .utils.parameters import Parameters

__all__ = ['DeepVO', 'Parameters']