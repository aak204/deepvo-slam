#!/usr/bin/env python3
"""
Скрипт для обучения модели DeepVO

Использование:
    python scripts/train_deepvo.py
    python scripts/train_deepvo.py --fine_tune
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train import main

if __name__ == '__main__':
    main()