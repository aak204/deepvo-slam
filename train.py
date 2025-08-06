"""
Скрипт обучения модели DeepVO

Поддерживает как обучение с нуля, так и fine-tuning предобученной модели.
Использует CosineAnnealingLR scheduler и сохраняет лучшую модель по validation loss.
"""

import argparse
import os
import time
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from deepvo.slam.deepvo_model import DeepVO
from utils.data import ImageSequenceDataset
from deepvo.utils.parameters import Parameters


def main():
    """Основная функция обучения."""
    # Обработка аргументов командной строки
    parser = argparse.ArgumentParser(description='DeepVO Training/Fine-tuning')
    parser.add_argument('--fine_tune', action='store_true',
                       help='Активировать режим fine-tuning.')
    args = parser.parse_args()
    
    # Создаем параметры с учетом режима fine-tuning
    par = Parameters(fine_tuning_mode=args.fine_tune)

    # Запись логов и создание датасетов
    mode = 'a' if par.resume else 'w'
    with open(par.record_path, mode) as f:
        f.write('\n' + '=' * 50 + '\n')
        f.write(f'Activated fine-tuning mode: {args.fine_tune}\n')
        f.write('\n'.join(f"{item[0]}: {item[1]}" for item in vars(par).items()))
        f.write('\n' + '=' * 50 + '\n')

    print(f"Fine-tuning mode: {args.fine_tune}")
    print(f"Epochs: {par.epochs}, LR: {par.optim_lr}, Experiment: {par.experiment_name}")

    # Создание датасетов
    train_dataset = ImageSequenceDataset(
        par.train_video, par.seq_len,
        drop_last=True, overlap=par.overlap
    )
    train_dl = DataLoader(
        train_dataset, batch_size=par.batch_size, shuffle=True,
        num_workers=par.n_processors, pin_memory=par.pin_mem
    )

    valid_dataset = ImageSequenceDataset(
        par.valid_video, par.seq_len,
        drop_last=True, overlap=1
    )
    valid_dl = DataLoader(
        valid_dataset, batch_size=par.batch_size, shuffle=True,
        num_workers=par.n_processors, pin_memory=par.pin_mem
    )

    print(f'Number of samples in training dataset: {len(train_dataset)}')
    print(f'Number of samples in validation dataset: {len(valid_dataset)}')
    print('=' * 50)

    # Создание модели и оптимизатора
    model = DeepVO(par.img_h, par.img_w, par.batch_norm)
    model.load_Flownet()
    model = model.to(par.device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=par.optim_lr,
        betas=(0.9, 0.999),
        weight_decay=par.optim_decay
    )

    print("Using CosineAnnealingLR scheduler.")
    scheduler = CosineAnnealingLR(optimizer, T_max=par.epochs, eta_min=1e-7)

    # Загрузка весов при необходимости
    if par.resume:
        if os.path.exists(par.load_model_path):
            model.load_state_dict(torch.load(par.load_model_path))
            print(f'Loaded model from: {par.load_model_path}')
        else:
            print(f"ERROR: Model file not found at {par.load_model_path}. "
                  f"Cannot fine-tune without pre-trained model!")
            return

        if os.path.exists(par.load_optimizer_path):
            optimizer.load_state_dict(torch.load(par.load_optimizer_path))
            print(f'Loaded optimizer from: {par.load_optimizer_path}')
        else:
            print(f"WARNING: Optimizer file not found at {par.load_optimizer_path}. "
                  f"Using new optimizer.")

    # Цикл обучения
    print(f'Record loss in: {par.record_path}')
    min_loss_valid = 1e10

    for ep in range(par.epochs):
        st_t = time.time()
        print('=' * 50)
        print(f'Epoch {ep + 1}/{par.epochs} | Current LR: {optimizer.param_groups[0]["lr"]:.2e}')

        # Обучение и валидация
        loss_mean_train, loss_ang_mean_train, loss_trans_mean_train = model.train_net(train_dl, optimizer)
        loss_mean_valid, loss_ang_mean_valid, loss_trans_mean_valid = model.valid_net(valid_dl)

        scheduler.step()

        # Логирование результатов
        train_message = (
            f'Epoch {ep + 1}\n  train loss: {loss_mean_train:.4f} '
            f'(ang: {loss_ang_mean_train:.4f}, trans: {loss_trans_mean_train:.4f}), '
            f'time: {time.time()-st_t:.2f}s'
        )
        print(train_message)
        with open(par.record_path, 'a') as f:
            f.write(train_message + '\n')

        valid_message = (
            f'  valid loss: {loss_mean_valid:.4f} '
            f'(ang: {loss_ang_mean_valid:.4f}, trans: {loss_trans_mean_valid:.4f})'
        )
        print(valid_message)
        with open(par.record_path, 'a') as f:
            f.write(valid_message + '\n')

        # Сохранение лучшей модели
        if loss_mean_valid < min_loss_valid:
            min_loss_valid = loss_mean_valid
            print(f'  -> New best validation loss: {min_loss_valid:.4f}. Saving model...')
            torch.save(model.state_dict(), par.save_model_path + '.best')
            torch.save(optimizer.state_dict(), par.save_optimzer_path + '.best')

        # Сохранение текущей модели
        torch.save(model.state_dict(), par.save_model_path + f'.epoch{ep+1}')
        torch.save(optimizer.state_dict(), par.save_optimzer_path + f'.epoch{ep+1}')
        torch.save(model.state_dict(), par.save_model_path + '.latest')
        torch.save(optimizer.state_dict(), par.save_optimzer_path + '.latest')

    print("Training completed!")


if __name__ == '__main__':
    main()