import glob
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset 
from deepvo.utils.parameters import par
import torchvision.transforms.functional as TF
import os
import torchvision.transforms as transforms
import random

# Generate the data frames per path
def get_data_info(folder_list, seq_len, drop_last, overlap):
    X_path, Y = [], []
    
    for folder in folder_list:
        pose_path = os.path.join(par.pose_dir, f'{folder}.npy')
        # Проверка, существует ли файл поз
        if not os.path.exists(pose_path):
            print(f"Warning: Pose file not found at {pose_path}. Skipping sequence {folder}.")
            continue
            
        poses = np.load(pose_path)  # (n_images, 6)
        img_dir_path = os.path.join(par.image_dir, folder, 'image_2/*.png')
        fpaths = glob.glob(img_dir_path)       
        fpaths.sort()
    
        n_frames = len(fpaths)
        start = 0
        while start + seq_len <= n_frames: # <= чтобы включить последний полный сэмпл
            x_seg = fpaths[start:start+seq_len]
            # Позы соответствуют смещению МЕЖДУ кадрами, поэтому их на 1 меньше
            Y.append(poses[start:start+seq_len-1])
            X_path.append(x_seg)
            
            if seq_len - overlap <= 0:
                raise ValueError("overlap must be less than seq_len")
            start += seq_len - overlap

        if not drop_last and start < n_frames:
             # Обработка последнего "хвоста", если он есть
            x_seg = fpaths[start:]
            y_seg = poses[start:min(start+len(x_seg)-1, len(poses))] # Берем доступные позы
            if len(x_seg) > 1: # Добавляем только если есть хотя бы 2 кадра
                X_path.append(x_seg)
                Y.append(y_seg)
    
    data = {'image_path': X_path, 'pose': Y}
    return data


class ImageSequenceDataset(Dataset):
    def __init__(self, image_folders, seq_len, drop_last, overlap):
        self.data_info = get_data_info(image_folders, seq_len, drop_last, overlap) 
        self.image_paths = self.data_info['image_path']
        self.groundtruth = self.data_info['pose']
        
        self.color_jitter_transform = None
        if par.is_color_jitter:
            self.color_jitter_transform = transforms.ColorJitter(
                brightness=par.brightness,
                contrast=par.contrast,
                saturation=par.saturation,
                hue=par.hue
            )
    
    def __getitem__(self, index):
        flag_hflip = par.is_hflip and random.random() > 0.5
        
        image_path_sequence = self.image_paths[index]
        image_sequence = []
        for img_path in image_path_sequence:
            img_as_img = Image.open(img_path)
            
            # --- НАЧАЛО ИЗМЕНЕНИЙ: КОНВЕРТАЦИЯ RGB -> BGR ---
            
            # 1. FlowNet ожидает BGR, а PIL загружает RGB. Меняем каналы местами.
            # numpy_image shape: (H, W, 3) with RGB
            numpy_image = np.array(img_as_img) 
            # Меняем местами 0-й и 2-й каналы: (H, W, 3) with BGR
            bgr_image = numpy_image[:, :, ::-1] 
            # Конвертируем обратно в PIL Image для дальнейших трансформаций
            img_as_img = Image.fromarray(bgr_image)

            # --- КОНЕЦ ИЗМЕНЕНИЙ ---

            # 2. Фотометрическая аугментация
            if self.color_jitter_transform:
                img_as_img = self.color_jitter_transform(img_as_img)
            
            # 3. Изменение размера
            img_as_img = TF.resize(img_as_img, size=(par.img_h, par.img_w))
            
            # 4. Горизонтальное отражение
            if flag_hflip:
                img_as_img = TF.hflip(img_as_img)
            
            # 5. Конвертация в тензор и нормализация [-0.5, 0.5] для FlowNet
            img_as_tensor = TF.to_tensor(img_as_img) - 0.5
            
            image_sequence.append(img_as_tensor)
        
        image_sequence = torch.stack(image_sequence, 0)
        gt_sequence_np = self.groundtruth[index]
        current_seq_len = image_sequence.shape[0]
        if gt_sequence_np.shape[0] != current_seq_len - 1:
            padded_gt = np.zeros((current_seq_len - 1, 6), dtype=np.float32)
            valid_len = min(padded_gt.shape[0], gt_sequence_np.shape[0])
            padded_gt[:valid_len, :] = gt_sequence_np[:valid_len, :]
            gt_sequence = torch.from_numpy(padded_gt)
        else:
            gt_sequence = torch.from_numpy(gt_sequence_np.astype(np.float32))

        if flag_hflip:
            gt_sequence[:, 0] *= -1.0
            gt_sequence[:, 2] *= -1.0
            gt_sequence[:, 3] *= -1.0
                
        return image_sequence, gt_sequence

    def __len__(self):
        return len(self.image_paths)

    def __len__(self):
        return len(self.image_paths)