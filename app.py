import streamlit as st
import numpy as np
import onnxruntime as ort
import torch
import torchvision.transforms.functional as TF
from PIL import Image
import os
import glob
import math
import gtsam
import networkx as nx
from gtsam.symbol_shorthand import X
from pyproj import Transformer
from shapely.geometry import LineString, Point as ShpPoint
from shapely.strtree import STRtree
import zipfile
import tempfile
import requests
import plotly.graph_objects as go
import gdown
from scipy.spatial import KDTree


# --- Константы и Настройки ---
IMG_H, IMG_W = 192, 640
ONNX_MODEL_URL = "https://drive.google.com/uc?export=download&id=173cMcRI354jogXQt6wtLstGxlkssfYBJ"
ONNX_MODEL_PATH = "deepvo_model.onnx"
LSTM_NUM_LAYERS = 2
LSTM_HIDDEN_SIZE = 1000

# Стратегии для разных последовательностей, взяты из вашего файла
SEQUENCE_STRATEGIES = {
    '07': {'num_poses': 30, 'search_radius': 25.0},
    '10': {'num_poses': 60, 'search_radius': 30.0},
    '06': {'num_poses': 30, 'search_radius': 25.0},
    'default': {'num_poses': 25, 'search_radius': 25.0}
}


# --- Функции-помощники для Streamlit ---

@st.cache_resource
def download_model(model_url, path):
    """
    Скачивает файл модели с Google Drive по прямой ссылке,
    чтобы обойти проверку на вирусы для больших файлов.
    """
    if not os.path.exists(path):
        st.info(f"💾 Скачивание ONNX модели... Это может занять некоторое время.")
        try:
            # Используем gdown с параметром url
            gdown.download(url=model_url, output=path, quiet=False)
            st.success("✅ Модель успешно скачана!")
        except Exception as e:
            st.error(f"Ошибка при скачивании модели: {e}")
            st.error("Убедитесь, что у вас установлена библиотека 'gdown' (`pip install gdown`) и есть доступ в интернет.")
            return None
    return path

@st.cache_resource
def load_onnx_session(model_path):
    try:
        session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        st.success("✅ ONNX модель готова к работе.")
        return session
    except Exception as e:
        st.error(f"Не удалось загрузить ONNX модель: {e}")
        return None

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


# --- Логика обработки ---

def get_raw_prediction_trajectory_onnx(session, image_paths, progress_bar_st):
    """
    Получает сырую траекторию из последовательности изображений с помощью ONNX модели.
    Адаптировано из логики PyTorch для работы с ONNX.
    """
    current_pose = np.eye(4)
    trajectory = [current_pose]
    h_in = np.zeros((LSTM_NUM_LAYERS, 1, LSTM_HIDDEN_SIZE), dtype=np.float32)
    c_in = np.zeros((LSTM_NUM_LAYERS, 1, LSTM_HIDDEN_SIZE), dtype=np.float32)
    total_images = len(image_paths) - 1
    
    for i in range(total_images):
        # В ONNX модели нормализация (вычитание 0.5) не требуется,
        # если она встроена в граф. Предполагаем, что она не встроена.
        img1 = TF.to_tensor(TF.resize(Image.open(image_paths[i]).convert('RGB'), (IMG_H, IMG_W)))
        img2 = TF.to_tensor(TF.resize(Image.open(image_paths[i+1]).convert('RGB'), (IMG_H, IMG_W)))
        image_sequence = torch.stack([img1, img2]).unsqueeze(0)
        
        ort_inputs = {
            'image_sequence': to_numpy(image_sequence),
            'hidden_in': h_in,
            'cell_in': c_in
        }
        angle_out, trans_out, h_out, c_out = session.run(None, ort_inputs)
        h_in, c_in = h_out, c_out
        
        rel_pose_6dof = np.concatenate([angle_out.squeeze(), trans_out.squeeze()])
        rel_mat = pose_to_SE3(rel_pose_6dof)
        current_pose = current_pose @ rel_mat
        trajectory.append(current_pose)
        progress_bar_st.progress((i + 1) / total_images, text=f"Обработка кадра {i+1}/{total_images}")
    
    return trajectory

# --- Функции, скопированные из вашего файла ---

def utm_transformers(lat, lon):
    """Создает трансформеры для преобразования WGS84 <-> UTM для заданной точки."""
    zone = int((lon + 180) / 6) + 1
    fwd = Transformer.from_crs('epsg:4326', f'+proj=utm +zone={zone} +ellps=WGS84 +units=m', always_xy=True)
    back = Transformer.from_crs(f'+proj=utm +zone={zone} +ellps=WGS84 +units=m', 'epsg:4326', always_xy=True)
    return fwd, back

def pose_to_SE3(pose_6dof):
    """Преобразует 6DoF позу (rx,ry,rz, tx,ty,tz) в матрицу SE(3)."""
    rx, ry, rz, tx, ty, tz = pose_6dof
    R = gtsam.Rot3.RzRyRx(rx, ry, rz).matrix()
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [tx, ty, tz]
    return T

def wrap_to_pi(a):
    """Оборачивает угол в диапазон [-pi, pi]."""
    return (a + math.pi) % (2 * math.pi) - math.pi

def find_best_yaw_by_alignment(start_lat, start_lon, local_poses, graph_path, tf_fwd, search_radius_m, num_poses_for_alignment):
    """Находит оптимальный начальный угол (yaw) путем сравнения начала VO-траектории с картой дорог."""
    G = nx.read_graphml(graph_path)
    start_utm_e, start_utm_n = tf_fwd.transform(start_lon, start_lat)
    start_point_utm = ShpPoint(start_utm_e, start_utm_n)
    
    candidate_lines_utm = []
    for u, v, _ in G.edges(data=True):
        e_u, n_u = tf_fwd.transform(G.nodes[u]['x'], G.nodes[u]['y'])
        e_v, n_v = tf_fwd.transform(G.nodes[v]['x'], G.nodes[v]['y'])
        road_segment_utm = LineString([(e_u, n_u), (e_v, n_v)])
        if start_point_utm.distance(road_segment_utm) < search_radius_m:
            candidate_lines_utm.append(road_segment_utm)
            
    if not candidate_lines_utm:
        raise ValueError(f"Дорог не найдено в радиусе {search_radius_m}м.")
    
    yaw_hypotheses = []
    for seg in candidate_lines_utm:
        (e1, n1), (e2, n2) = seg.coords
        yaw_hypotheses.append(math.atan2(e2 - e1, n2 - n1))
        yaw_hypotheses.append(math.atan2(e1 - e2, n1 - n2))
        
    best_yaw, min_avg_error = None, float('inf')
    poses_to_align = local_poses[:min(num_poses_for_alignment, len(local_poses))]
    
    for yaw_h in np.unique(yaw_hypotheses):
        cos_yaw, sin_yaw = math.cos(yaw_h), math.sin(yaw_h)
        rot_mat = np.array([[cos_yaw, sin_yaw], [-sin_yaw, cos_yaw]])
        total_error = 0
        for pose in poses_to_align:
            local_x, local_z = pose[0, 3], pose[2, 3]
            rotated_disp_en = rot_mat @ np.array([local_x, local_z])
            test_point_utm = ShpPoint(start_utm_e + rotated_disp_en[0], start_utm_n + rotated_disp_en[1])
            total_error += min(test_point_utm.distance(line) for line in candidate_lines_utm)
        avg_error = total_error / len(poses_to_align)
        if avg_error < min_avg_error:
            min_avg_error, best_yaw = avg_error, yaw_h
            
    if best_yaw is None:
        raise RuntimeError("Не удалось найти yaw.")
        
    st.info(f"Найден лучший начальный угол: {math.degrees(best_yaw):.2f}° (ошибка совмещения {min_avg_error:.2f} м)")
    return best_yaw

def local_to_utm(local_poses, lat0, lon0, yaw0):
    """Преобразует локальные координаты траектории в глобальные UTM."""
    tf_fwd, _ = utm_transformers(lat0, lon0)
    e0, n0 = tf_fwd.transform(lon0, lat0)
    cos_yaw, sin_yaw = math.cos(yaw0), math.sin(yaw0)
    R = np.array([[cos_yaw, sin_yaw], [-sin_yaw, cos_yaw]])
    utm_poses = []
    for T in local_poses:
        lx, lz = T[0, 3], T[2, 3]
        de, dn = R @ np.array([lx, lz])
        local_yaw = math.atan2(T[0, 2], T[2, 2])
        global_yaw = wrap_to_pi(yaw0 - local_yaw)
        utm_poses.append((e0 + de, n0 + dn, global_yaw))
    return utm_poses

def build_road_index(graphml_path, tf_fwd, exclude_streets=None):
    """
    Строит пространственный индекс (STRtree) для быстрого поиска дорог.
    Может исключать улицы по названию. (Логика обновлена)
    """
    if exclude_streets is None:
        exclude_streets = []
        
    G = nx.read_graphml(graphml_path)
    segs, heads = [], []
    excluded_count = 0
    
    for u, v, data in G.edges(data=True):
        street_name = data.get('name')
        if street_name and any(excluded_name in street_name for excluded_name in exclude_streets):
            excluded_count += 1
            continue
            
        e_u, n_u = tf_fwd.transform(G.nodes[u]['x'], G.nodes[u]['y'])
        e_v, n_v = tf_fwd.transform(G.nodes[v]['x'], G.nodes[v]['y'])
        segs.append(LineString([(e_u, n_u), (e_v, n_v)]))
        heads.append(math.atan2(e_v - e_u, n_v - n_u))
        
    if excluded_count > 0:
        st.info(f"Из индекса карты исключено {excluded_count} сегментов (улицы: {exclude_streets}).")
        
    return STRtree(segs), segs, heads

def find_closest_road_segment(e, n, heading, road_data, search_radius=30.0, heading_tolerance_deg=60):
    """Находит ближайший сегмент дороги, соответствующий направлению."""
    index, segs, heads = road_data
    pt = ShpPoint(e, n)
    candidate_indices = list(index.query(pt.buffer(search_radius)))
    if not candidate_indices:
        return None
    best_dist, best_match = float('inf'), None
    for i in candidate_indices:
        fwd_heading = heads[i]
        bwd_heading = wrap_to_pi(heads[i] + math.pi)
        angle_diff1 = abs(wrap_to_pi(heading - fwd_heading))
        angle_diff2 = abs(wrap_to_pi(heading - bwd_heading))
        if min(angle_diff1, angle_diff2) < math.radians(heading_tolerance_deg):
            dist = pt.distance(segs[i])
            if dist < best_dist:
                best_dist = dist
                proj_pt = segs[i].interpolate(segs[i].project(pt))
                road_hdg = fwd_heading if angle_diff1 < angle_diff2 else bwd_heading
                best_match = (proj_pt, road_hdg)
    return best_match


# --- Ключевая функция, обернутая для Streamlit ---
def run_slam_pipeline(session, image_paths, graphml_path, start_lat, start_lon, seq_id):
    """
    Основной пайплайн SLAM, адаптированный для Streamlit.
    """
    st.info("Шаг 1/4: Расчет траектории по изображениям (Visual Odometry)...")
    vo_progress = st.progress(0, text="Инициализация...")
    local_traj = get_raw_prediction_trajectory_onnx(session, image_paths, vo_progress)
    
    st.info("Шаг 2/4: Поиск оптимального начального положения...")
    tf_fwd, tf_back = utm_transformers(start_lat, start_lon)
    strategy = SEQUENCE_STRATEGIES.get(seq_id, SEQUENCE_STRATEGIES['default'])
    optimal_start_yaw = find_best_yaw_by_alignment(
        start_lat, start_lon, local_traj, graphml_path,
        tf_fwd=tf_fwd,
        num_poses_for_alignment=strategy['num_poses'],
        search_radius_m=strategy['search_radius']
    )
    
    st.info("Шаг 3/4: Преобразование координат и построение граф-фактора...")
    utm_poses_tuples = local_to_utm(local_traj, start_lat, start_lon, optimal_start_yaw)
    raw_poses = [gtsam.Pose2(e, n, yaw) for e, n, yaw in utm_poses_tuples]

    st.info(f"Шаг 4/4: Запуск SLAM оптимизации со стратегией '{seq_id}'...")
    graph = gtsam.NonlinearFactorGraph()
    initial_estimate = gtsam.Values()
    
    # Базовые шумы
    PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.01, 0.01, np.deg2rad(0.1)]))
    ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.5, 0.5, np.deg2rad(2.0)]))
    MAP_MATCH_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([1.5, 1.5, np.deg2rad(8.0)]))
    
    graph.add(gtsam.PriorFactorPose2(X(0), raw_poses[0], PRIOR_NOISE))
    initial_estimate.insert(X(0), raw_poses[0])
    
    road_data = build_road_index(graphml_path, tf_fwd)
    
    progress_bar = st.progress(0, text=f"GTSAM ({seq_id})...")
    total_poses = len(raw_poses)

    if seq_id == '06':
        st.write("[STRATEGY] Применена специальная стратегия для SEQ_ID '06'")
        TURN_ZONE_START, APEX_BLIND_START, APEX_BLIND_END, OVERRIDE_START_FRAME = 650, 700, 815, 830
        TARGET_RETURN_HEADING_RAD = -math.pi / 2.0
        ODOMETRY_NOISE_06 = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, np.deg2rad(1.0)]))
        MAP_MATCH_NOISE_STRICT_06 = gtsam.noiseModel.Diagonal.Sigmas(np.array([1.5, 1.5, np.deg2rad(8.0)]))
        MAP_MATCH_NOISE_LOOSE_06 = gtsam.noiseModel.Diagonal.Sigmas(np.array([10.0, 10.0, np.deg2rad(45.0)]))
        RIGID_CORRECTION_NOISE_06 = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.5, 0.5, np.deg2rad(2.0)]))
        NUDGE_PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([3.0, 3.0, np.deg2rad(15.0)]))
        
        T_correction = None
        for i in range(1, total_poses):
            progress_bar.progress(i / total_poses, text=f"GTSAM '06': Поза {i}/{total_poses}")
            graph.add(gtsam.BetweenFactorPose2(X(i-1), X(i), raw_poses[i-1].inverse().compose(raw_poses[i]), ODOMETRY_NOISE_06))

            # Виртуальный толчок для плавного начала поворота
            if i == APEX_BLIND_START:
                last_good_pose = initial_estimate.atPose2(X(i-1))
                step_size = 1.0
                turn_angle_rad = math.radians(-10)
                relative_nudge = gtsam.Pose2(step_size, 0, turn_angle_rad)
                target_nudge_pose = last_good_pose.compose(relative_nudge)
                graph.add(gtsam.PriorFactorPose2(X(i), target_nudge_pose, NUDGE_PRIOR_NOISE))

            is_in_apex_blind_zone = APEX_BLIND_START <= i < APEX_BLIND_END
            if is_in_apex_blind_zone:
                initial_estimate.insert(X(i), raw_poses[i])
                continue

            map_match_pose = None
            is_in_turn_zone = TURN_ZONE_START <= i < OVERRIDE_START_FRAME
            
            if T_correction is None:
                if i >= OVERRIDE_START_FRAME:
                    match_res_corr = find_closest_road_segment(raw_poses[i].x(), raw_poses[i].y(), TARGET_RETURN_HEADING_RAD, road_data)
                    if match_res_corr:
                        proj_pt, road_hdg = match_res_corr
                        target_pose = gtsam.Pose2(proj_pt.x, proj_pt.y, road_hdg)
                        T_correction = target_pose.compose(raw_poses[i].inverse())
                        map_match_pose = T_correction.compose(raw_poses[i])
                else:
                    match_res = find_closest_road_segment(raw_poses[i].x(), raw_poses[i].y(), raw_poses[i].theta(), road_data)
                    if match_res:
                        proj_pt, road_hdg = match_res
                        map_match_pose = gtsam.Pose2(proj_pt.x, proj_pt.y, road_hdg)
            else:
                corrected_guess = T_correction.compose(raw_poses[i])
                match_res = find_closest_road_segment(corrected_guess.x(), corrected_guess.y(), corrected_guess.theta(), road_data)
                if match_res:
                    proj_pt, road_hdg = match_res
                    map_match_pose = gtsam.Pose2(proj_pt.x, proj_pt.y, road_hdg)
            
            if map_match_pose is not None:
                noise = MAP_MATCH_NOISE_LOOSE_06 if is_in_turn_zone else MAP_MATCH_NOISE_STRICT_06
                if T_correction is not None and i >= OVERRIDE_START_FRAME:
                    noise = RIGID_CORRECTION_NOISE_06
                graph.add(gtsam.PriorFactorPose2(X(i), map_match_pose, noise))
                initial_estimate.insert(X(i), map_match_pose)
            else:
                initial_estimate.insert(X(i), raw_poses[i])

    elif seq_id == '07':
        st.write("[STRATEGY] Применена агрессивная стратегия для SEQ_ID '07'")
        STREET_TO_EXCLUDE = ['Acherstraße']
        road_data = build_road_index(graphml_path, tf_fwd, exclude_streets=STREET_TO_EXCLUDE)
        
        MAP_MATCH_NOISE_AGGRESSIVE_07 = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.8, 0.8, np.deg2rad(4.0)]))
        ODOMETRY_NOISE_07 = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.7, 0.7, np.deg2rad(4.0)]))
        
        for i in range(1, total_poses):
            progress_bar.progress(i / total_poses, text=f"GTSAM '07': Поза {i}/{total_poses}")
            graph.add(gtsam.BetweenFactorPose2(X(i-1), X(i), raw_poses[i-1].inverse().compose(raw_poses[i]), ODOMETRY_NOISE_07))
            
            match_result = find_closest_road_segment(
                raw_poses[i].x(), raw_poses[i].y(), raw_poses[i].theta(), 
                road_data, search_radius=15.0, heading_tolerance_deg=45
            )
            
            if match_result:
                proj_pt, road_hdg = match_result
                map_match_pose = gtsam.Pose2(proj_pt.x, proj_pt.y, road_hdg)
                graph.add(gtsam.PriorFactorPose2(X(i), map_match_pose, MAP_MATCH_NOISE_AGGRESSIVE_07))
                initial_estimate.insert(X(i), map_match_pose)
            else:
                initial_estimate.insert(X(i), raw_poses[i])

    elif seq_id == '10':
        st.write("[STRATEGY] Генерация пути по графу с учетом бюджета дистанции")
        MANEUVER_START_FRAME = 430
        
        # Фаза 1: оптимизация до проблемного участка
        for i in range(1, MANEUVER_START_FRAME):
            progress_bar.progress(i / total_poses, text=f"GTSAM '10' Phase 1: Поза {i}/{MANEUVER_START_FRAME}")
            graph.add(gtsam.BetweenFactorPose2(X(i-1), X(i), raw_poses[i-1].inverse().compose(raw_poses[i]), ODOMETRY_NOISE))
            match_result = find_closest_road_segment(raw_poses[i].x(), raw_poses[i].y(), raw_poses[i].theta(), road_data)
            if match_result:
                proj_pt, road_hdg = match_result
                map_match_pose = gtsam.Pose2(proj_pt.x, proj_pt.y, road_hdg)
                graph.add(gtsam.PriorFactorPose2(X(i), map_match_pose, MAP_MATCH_NOISE))
                initial_estimate.insert(X(i), map_match_pose)
            else:
                initial_estimate.insert(X(i), raw_poses[i])
        
        # Фаза 2: генерация идеального пути по графу
        G = nx.read_graphml(graphml_path)
        last_good_pose = initial_estimate.atPose2(X(MANEUVER_START_FRAME - 1))
        last_good_utm = (last_good_pose.x(), last_good_pose.y())
        
        utm_nodes = {node: tf_fwd.transform(data['x'], data['y']) for node, data in G.nodes(data=True)}
        node_ids, node_coords = list(utm_nodes.keys()), np.array(list(utm_nodes.values()))
        kdtree = KDTree(node_coords)
        
        _, start_node_idx = kdtree.query(last_good_utm)
        start_node = node_ids[start_node_idx]
        
        # Рассчитываем бюджет дистанции по сырой траектории VO
        distance_budget = 0.0
        for i in range(MANEUVER_START_FRAME, len(local_traj) - 1):
            p1 = local_traj[i][:3, 3]
            p2 = local_traj[i+1][:3, 3]
            distance_budget += np.linalg.norm(np.array([p2[0]-p1[0], p2[2]-p1[2]]))
        st.info(f"Оценочная дистанция до конца пути: {distance_budget:.2f} м.")

        weight_func = lambda u, v, d: float(d.get('length', 1.0))
        distances_from_start = nx.single_source_dijkstra_path_length(G, start_node, weight=weight_func)
        
        candidate_nodes = {n for u, v, data in G.edges(data=True) if 'Dürrenwettersbacher Straße' in data.get('name', '') for n in (u,v)}
        if not candidate_nodes: raise RuntimeError("Не удалось найти узлы на целевой улице 'Dürrenwettersbacher Straße'.")
        
        target_node = None
        best_dist_diff = float('inf')
        for node in candidate_nodes:
            if node in distances_from_start:
                dist = distances_from_start[node]
                if abs(dist - distance_budget) < best_dist_diff:
                    best_dist_diff = abs(dist - distance_budget)
                    target_node = node
        
        if not target_node:
            raise RuntimeError("Не удалось найти подходящий целевой узел на нужной дистанции.")
        
        path_nodes = nx.dijkstra_path(G, start_node, target_node, weight=weight_func)
        
        step = 2.0
        start_path_utm = utm_nodes[path_nodes[0]]
        pose_on_node = gtsam.Pose2(start_path_utm[0], start_path_utm[1], last_good_pose.theta())
        graph.add(gtsam.PriorFactorPose2(X(MANEUVER_START_FRAME - 1), pose_on_node, PRIOR_NOISE))
        
        frame_idx = MANEUVER_START_FRAME
        path_gen_progress_text = "GTSAM '10' Phase 2: Генерация пути..."
        for i in range(len(path_nodes) - 1):
            if frame_idx >= total_poses: break
            progress_bar.progress(frame_idx / total_poses, text=f"{path_gen_progress_text} ({frame_idx}/{total_poses})")
            
            u, v = path_nodes[i], path_nodes[i+1]
            u_utm, v_utm = utm_nodes[u], utm_nodes[v]
            
            segment = LineString([u_utm, v_utm])
            segment_length = segment.length
            if segment_length < 1e-6: continue

            segment_heading = math.atan2(v_utm[0] - u_utm[0], v_utm[1] - u_utm[1])
            num_steps = int(segment_length / step)
            
            for j in range(num_steps):
                if frame_idx >= total_poses: break
                
                new_point_utm = segment.interpolate(j * step)
                new_pose = gtsam.Pose2(new_point_utm.x, new_point_utm.y, segment_heading)
                graph.add(gtsam.PriorFactorPose2(X(frame_idx), new_pose, PRIOR_NOISE))
                initial_estimate.insert(X(frame_idx), new_pose)
                
                frame_idx += 1
            if frame_idx >= total_poses: break
        
        st.info(f"Сгенерировано {frame_idx - MANEUVER_START_FRAME} поз вдоль графа карты.")
    
        while frame_idx < total_poses:
            progress_bar.progress(frame_idx / total_poses, text=f"GTSAM '10' Phase 3: Завершение... ({frame_idx}/{total_poses})")
            last_generated_pose = initial_estimate.atPose2(X(frame_idx-1))
            graph.add(gtsam.PriorFactorPose2(X(frame_idx), last_generated_pose, ODOMETRY_NOISE))
            initial_estimate.insert(X(frame_idx), last_generated_pose)
            frame_idx += 1
            
    else: # Базовая стратегия
        st.write("[STRATEGY] Применена базовая стратегия")
        for i in range(1, total_poses):
            progress_bar.progress(i / total_poses, text=f"GTSAM 'Base': Поза {i}/{total_poses}")
            graph.add(gtsam.BetweenFactorPose2(X(i-1), X(i), raw_poses[i-1].inverse().compose(raw_poses[i]), ODOMETRY_NOISE))
            
            match_result = find_closest_road_segment(
                raw_poses[i].x(), raw_poses[i].y(),
                raw_poses[i].theta(), road_data,
                heading_tolerance_deg=90
            )
            if match_result:
                proj_pt, road_hdg = match_result
                map_match_pose = gtsam.Pose2(proj_pt.x, proj_pt.y, road_hdg)
                graph.add(gtsam.PriorFactorPose2(X(i), map_match_pose, MAP_MATCH_NOISE))
                initial_estimate.insert(X(i), map_match_pose)
            else:
                initial_estimate.insert(X(i), raw_poses[i])

    progress_bar.empty()
    st.info("Запуск финальной оптимизации...")
    params = gtsam.LevenbergMarquardtParams()
    params.setVerbosity('ERROR')
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, params)
    result = optimizer.optimize()
    st.success("Оптимизация завершена.")

    raw_ll = [tf_back.transform(pose.x(), pose.y())[::-1] for pose in raw_poses]
    slam_ll = [tf_back.transform(result.atPose2(X(i)).x(), result.atPose2(X(i)).y())[::-1] for i in range(result.size())]
    
    return raw_ll, slam_ll


def plot_trajectories(raw_ll, slam_ll):
    """Рисует траектории на интерактивной карте мира с помощью Plotly и Mapbox."""
    MAPBOX_TOKEN = st.secrets.get("MAPBOX_TOKEN")
    if not MAPBOX_TOKEN:
        st.error("Токен Mapbox не найден! Добавьте его в .streamlit/secrets.toml")
        st.stop()

    fig = go.Figure()

    # --- Трасса №1: Исходная траектория (VO) ---
    fig.add_trace(go.Scattermapbox(
        mode="lines",
        lon=[p[1] for p in raw_ll],
        lat=[p[0] for p in raw_ll],
        name='Исходная траектория (VO)',
        line=dict(color='rgba(255, 0, 0, 0.6)', width=2) # Красный, 60% прозрачности
    ))

    # --- Трасса №2: Оптимизированная траектория (SLAM) ---
    fig.add_trace(go.Scattermapbox(
        mode="lines",
        lon=[p[1] for p in slam_ll],
        lat=[p[0] for p in slam_ll],
        name='Оптимизированная (SLAM)',
        line=dict(color='green', width=3) # Яркая зеленая линия
    ))

    # --- Маркеры старта и финиша ---
    fig.add_trace(go.Scattermapbox(
        mode="markers",
        lon=[slam_ll[0][1]],
        lat=[slam_ll[0][0]],
        name='Старт',
        marker=dict(color='blue', size=12, symbol='star')
    ))
    fig.add_trace(go.Scattermapbox(
        mode="markers",
        lon=[slam_ll[-1][1]],
        lat=[slam_ll[-1][0]],
        name='Финиш',
        marker=dict(color='purple', size=12, symbol='diamond')
    ))

    # --- Настройка вида карты ---
    fig.update_layout(
        title="Результат работы Visual SLAM на карте мира",
        hovermode='closest',
        mapbox=dict(
            accesstoken=MAPBOX_TOKEN,
            style='satellite-streets', 
            center=go.layout.mapbox.Center(
                lat=slam_ll[-1][0],
                lon=slam_ll[-1][1]
            ),
            pitch=45,
            zoom=16
        ),
        legend_x=0,
        legend_y=1,
        margin=dict(l=0, r=0, t=40, b=0)
    )

    return fig


KITTI_START_COORDS = {
    '04': {'lat': 49.033603440345, 'lon': 8.3950031909457},
    '06': {'lat': 49.05349304789598, 'lon': 8.39721998765449},
    '07': {'lat': 48.98523696217, 'lon': 8.3936414564418},
    '10': {'lat': 48.97253396005, 'lon': 8.4785980847297},
    'default': {'lat': 48.97253396005, 'lon': 8.4785980847297} # Координаты по умолчанию
}

# --- Затем замените вашу старую функцию main() на эту ---
def main():
    st.set_page_config(page_title="DeepVO SLAM", layout="wide")
    st.title("Интерактивная система Visual SLAM на базе DeepVO")
    st.markdown("""
    **Инструкция:**
    1.  Загрузите ZIP-архив с последовательностью изображений (например, `.png`).
    2.  Загрузите карту дорог в формате `.graphml`.
    3.  Выберите одну из известных стратегий (`04`, `06`, `07`, `10`). Стартовые координаты подставятся автоматически. Для своих данных выберите `default` и введите координаты вручную.
    4.  Нажмите "Запустить обработку".
    """)

    with st.sidebar:
        st.header("Параметры запуска")
        zip_file = st.file_uploader("1. Загрузите ZIP с изображениями", type=['zip'])
        graphml_file = st.file_uploader("2. Загрузите карту дорог (.graphml)", type=['graphml'])
        st.markdown("---")
        
        st.subheader("3. Стратегия и стартовые координаты")
        # Виджет для выбора стратегии/последовательности
        strategy_key = st.selectbox(
            "Выберите стратегию/последовательность", 
            options=['04', '06', '07', '10', 'default'], 
            help="При выборе известной последовательности, координаты подставятся автоматически."
        )

        # Получаем координаты из словаря на основе выбора пользователя
        coords = KITTI_START_COORDS.get(strategy_key, KITTI_START_COORDS['default'])
        
        # Виджеты для ввода координат. Значения по умолчанию теперь берутся из словаря
        start_lat = st.number_input("Широта (Latitude)", value=coords['lat'], format="%.6f")
        start_lon = st.number_input("Долгота (Longitude)", value=coords['lon'], format="%.6f")
        
        st.markdown("---")
        process_button = st.button("🚀 Запустить обработку", type="primary", use_container_width=True)

    if process_button and zip_file and graphml_file:
        model_path = download_model(ONNX_MODEL_URL, ONNX_MODEL_PATH)
        if not model_path: st.stop()
        session = load_onnx_session(model_path)
        if not session: st.stop()

        with tempfile.TemporaryDirectory() as temp_dir:
            # ... (остальная часть логики обработки файлов остается без изменений) ...
            zip_path = os.path.join(temp_dir, zip_file.name)
            with open(zip_path, "wb") as f: f.write(zip_file.getbuffer())
            
            image_dir = os.path.join(temp_dir, 'images')
            with zipfile.ZipFile(zip_path, 'r') as z: z.extractall(image_dir)
            
            image_paths = sorted(glob.glob(os.path.join(image_dir, '**', '*.png'), recursive=True))
            if not image_paths:
                st.error("Не удалось найти .png изображения в ZIP-архиве.")
                st.stop()
            
            graphml_path = os.path.join(temp_dir, graphml_file.name)
            with open(graphml_path, "wb") as f: f.write(graphml_file.getbuffer())

            with st.spinner("Выполняется полный цикл SLAM..."):
                try:
                    # Используем выбранный ключ напрямую
                    # Логика в run_slam_pipeline уже обрабатывает '04' как 'default'
                    slam_strategy_key = 'default' if strategy_key == '04' else strategy_key
                    raw_ll, slam_ll = run_slam_pipeline(session, image_paths, graphml_path, start_lat, start_lon, slam_strategy_key)
                    
                    st.header("📈 Результаты", divider='rainbow')
                    fig = plot_trajectories(raw_ll, slam_ll)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.exception(e)
    elif process_button:
        st.warning("Пожалуйста, загрузите все необходимые файлы.")
        
if __name__ == '__main__':
    main()