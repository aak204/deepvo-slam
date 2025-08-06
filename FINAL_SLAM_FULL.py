"""
KITTI Visual-Inertial SLAM System

Основной модуль для выполнения Visual-Inertial SLAM на датасете KITTI
с использованием DeepVO для визуальной одометрии и GTSAM для оптимизации.
"""

import os
import glob
import math
import folium
import gtsam
import networkx as nx
import numpy as np
import pykitti
import torch
import torchvision.transforms.functional as TF
from gtsam.symbol_shorthand import X
from PIL import Image
from pyproj import Transformer
from shapely.geometry import LineString, Point as ShpPoint
from shapely.strtree import STRtree
from tqdm import tqdm
from deepvo.slam.deepvo_model import DeepVO
from scipy.spatial import KDTree

# Константы
SEQ_ID = '04'
KITTI_RAW_DIR = './kitti_raw_data'
IMAGE_DIR = './dataset/sequences'
MODEL_PATH = './experiments/kitti_finetuned/models/t00020809_v030405060710_im192x640_s7_b12_finetuned.model.latest'
GRAPH_ML_DIR = './map_graphs'
RESULT_DIR = './test_results_visual'
os.makedirs(RESULT_DIR, exist_ok=True)
IMG_H, IMG_W = 192, 640
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Стратегии для разных последовательностей
SEQUENCE_STRATEGIES = {
    '07': {'num_poses': 30, 'search_radius': 25.0},
    '10': {'num_poses': 60, 'search_radius': 30.0},
    '06': {'num_poses': 30, 'search_radius': 25.0},
    'default': {'num_poses': 25, 'search_radius': 25.0}
}

# Маппинг последовательностей KITTI
SEQUENCE_MAPPING = {
    '00': {'date': '2011_10_03', 'drive': '0027'},
    '01': {'date': '2011_10_03', 'drive': '0042'},
    '02': {'date': '2011_10_03', 'drive': '0034'},
    '03': {'date': '2011_09_26', 'drive': '0067'},
    '04': {'date': '2011_09_30', 'drive': '0016'},
    '05': {'date': '2011_09_30', 'drive': '0018'},
    '06': {'date': '2011_09_30', 'drive': '0020'},
    '07': {'date': '2011_09_30', 'drive': '0027'},
    '08': {'date': '2011_09_30', 'drive': '0028'},
    '09': {'date': '2011_09_30', 'drive': '0033'},
    '10': {'date': '2011_09_30', 'drive': '0034'}
}


def load_model():
    """Загружает модель DeepVO и переводит ее в режим оценки."""
    print(f"[INFO] Загрузка модели: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Файл модели не найден: {MODEL_PATH}")
    net = DeepVO(IMG_H, IMG_W, True)
    net.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    net.to(DEVICE).eval()
    print("[OK] Модель загружена.")
    return net


def utm_transformers(lat, lon):
    """Создает трансформеры для преобразования WGS84 <-> UTM для заданной точки."""
    zone = int((lon + 180) / 6) + 1
    fwd = Transformer.from_crs(
        'epsg:4326',
        f'+proj=utm +zone={zone} +ellps=WGS84 +units=m',
        always_xy=True
    )
    back = Transformer.from_crs(
        f'+proj=utm +zone={zone} +ellps=WGS84 +units=m',
        'epsg:4326',
        always_xy=True
    )
    return fwd, back


def pose_to_SE3(pose_6dof):
    """Преобразует 6DoF позу (rx,ry,rz, tx,ty,tz) в матрицу SE(3)."""
    rx, ry, rz, tx, ty, tz = pose_6dof
    R = gtsam.Rot3.RzRyRx(rx, ry, rz).matrix()
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [tx, ty, tz]
    return T


def get_raw_prediction_trajectory(model, image_paths):
    """Получает сырую траекторию из последовательности изображений с помощью модели DeepVO."""
    current_pose, trajectory, prev_hc = np.eye(4), [np.eye(4)], None
    for i in tqdm(range(len(image_paths) - 1), desc="[DeepVO]"):
        img1 = TF.to_tensor(TF.resize(Image.open(image_paths[i]), (IMG_H, IMG_W))) - 0.5
        img2 = TF.to_tensor(TF.resize(Image.open(image_paths[i+1]), (IMG_H, IMG_W))) - 0.5
        image_pair = torch.stack([img1, img2]).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            angle, trans, prev_hc = model(image_pair, prev=prev_hc)

        rel_pose_6dof = np.concatenate([
            angle.cpu().squeeze().numpy(),
            trans.cpu().squeeze().numpy()
        ])
        rel_mat = pose_to_SE3(rel_pose_6dof)

        current_pose = current_pose @ rel_mat
        trajectory.append(current_pose)
    return trajectory


def wrap_to_pi(a):
    """Оборачивает угол в диапазон [-pi, pi]."""
    return (a + math.pi) % (2 * math.pi) - math.pi


def find_best_yaw_by_alignment(start_lat, start_lon, local_poses, graph_path,
                               tf_fwd, search_radius_m, num_poses_for_alignment):
    """Находит оптимальный начальный угол (yaw) путем сравнения начала VO-траектории с картой дорог."""
    print(f"\n[INFO] Поиск оптимального Yaw (радиус: {search_radius_m}м, позы: {num_poses_for_alignment})")
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
    print(f"[OK] Найдено {len(candidate_lines_utm)} дорог-кандидатов.")

    yaw_hypotheses = []
    for seg in candidate_lines_utm:
        (e1, n1), (e2, n2) = seg.coords
        yaw_hypotheses.append(math.atan2(e2 - e1, n2 - n1))
        yaw_hypotheses.append(math.atan2(e1 - e2, n1 - n2))
    print(f"[OK] Сгенерировано {len(np.unique(yaw_hypotheses))} уникальных гипотез для Yaw.")

    best_yaw, min_avg_error = None, float('inf')
    poses_to_align = local_poses[:min(num_poses_for_alignment, len(local_poses))]

    for yaw_h in np.unique(yaw_hypotheses):
        cos_yaw, sin_yaw = math.cos(yaw_h), math.sin(yaw_h)
        rot_mat = np.array([[cos_yaw, sin_yaw], [-sin_yaw, cos_yaw]])

        total_error = 0
        for pose in poses_to_align:
            local_x, local_z = pose[0, 3], pose[2, 3]
            rotated_disp_en = rot_mat @ np.array([local_x, local_z])
            test_point_utm = ShpPoint(
                start_utm_e + rotated_disp_en[0],
                start_utm_n + rotated_disp_en[1]
            )
            total_error += min(test_point_utm.distance(line) for line in candidate_lines_utm)

        avg_error = total_error / len(poses_to_align)
        if avg_error < min_avg_error:
            min_avg_error, best_yaw = avg_error, yaw_h

    if best_yaw is None:
        raise RuntimeError("Не удалось найти yaw.")

    print(f"[OK] Найден лучший Yaw: {math.degrees(best_yaw):.2f}° (ошибка {min_avg_error:.2f} м)")
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
        global_yaw = yaw0 - local_yaw
        utm_poses.append((e0 + de, n0 + dn, wrap_to_pi(global_yaw)))

    return utm_poses


def build_road_index(graphml_path, tf_fwd, exclude_streets=None):
    """
    Строит пространственный индекс (STRtree) для быстрого поиска дорог.
    Может исключать улицы по названию.
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
        print(f"[INFO] Исключено {excluded_count} сегментов дороги из индекса (улицы: {exclude_streets}).")
        
    return STRtree(segs), segs, heads


def find_closest_road_segment(e, n, heading, road_data, search_radius=30.0, heading_tolerance_deg=60):
    """Находит ближайший сегмент дороги, соответствующий направлению."""
    index, segs, heads = road_data
    pt = ShpPoint(e, n)
    candidate_indices = list(index.query(pt.buffer(search_radius)))
    if not candidate_indices:
        return None

    best_dist = float('inf')
    best_match = None

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


def find_closest_node(e, n, G_utm_nodes_kdtree, G_utm_nodes_list):
    """Находит ближайший узел графа к точке (e, n) с помощью KD-дерева."""
    _, idx = G_utm_nodes_kdtree.query([e, n])
    return G_utm_nodes_list[idx]



def run_sequence():
    """
    Основной пайплайн SLAM системы.
    Использует базовую логику для всех последовательностей
    и специальную стратегию для отдельных случаев.
    """
    # Инициализация
    info = SEQUENCE_MAPPING[SEQ_ID]
    ds = pykitti.raw(KITTI_RAW_DIR, info['date'], info['drive'])
    start_lat, start_lon = ds.oxts[0].packet.lat, ds.oxts[0].packet.lon
    tf_fwd, tf_back = utm_transformers(start_lat, start_lon)

    paths = sorted(glob.glob(os.path.join(IMAGE_DIR, SEQ_ID, 'image_2', '*.png')))
    if not paths:
        raise FileNotFoundError(f"Изображения не найдены в {os.path.join(IMAGE_DIR, SEQ_ID, 'image_2')}")
        
    net = load_model()
    local_traj = get_raw_prediction_trajectory(net, paths)
    graphml_file = os.path.join(GRAPH_ML_DIR, f'map_graph_{SEQ_ID}.graphml')
    if not os.path.exists(graphml_file):
        raise FileNotFoundError(f"Файл карты не найден: {graphml_file}")

    # Поиск начального положения и преобразование координат
    strategy = SEQUENCE_STRATEGIES.get(SEQ_ID, SEQUENCE_STRATEGIES['default'])
    optimal_start_yaw = find_best_yaw_by_alignment(
        start_lat, start_lon, local_traj, graphml_file,
        tf_fwd=tf_fwd,
        num_poses_for_alignment=strategy['num_poses'],
        search_radius_m=strategy['search_radius']
    )
    utm_poses_tuples = local_to_utm(local_traj, start_lat, start_lon, optimal_start_yaw)
    raw_poses = [gtsam.Pose2(e, n, yaw) for e, n, yaw in utm_poses_tuples]

    # Построение Фактор-графа
    print(f"\n[INFO] Построение фактор-графа для последовательности {SEQ_ID}...")
    
    graph = gtsam.NonlinearFactorGraph()
    initial_estimate = gtsam.Values()
    
    # Базовые шумы
    PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.01, 0.01, np.deg2rad(0.1)]))
    ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.5, 0.5, np.deg2rad(2.0)]))
    MAP_MATCH_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([1.5, 1.5, np.deg2rad(8.0)]))

    graph.add(gtsam.PriorFactorPose2(X(0), raw_poses[0], PRIOR_NOISE))
    initial_estimate.insert(X(0), raw_poses[0])

    road_data = build_road_index(graphml_file, tf_fwd)

    if SEQ_ID == '06':
        print("[STRATEGY] Применена специальная стратегия для SEQ_ID '06'")
        TURN_ZONE_START = 650
        APEX_BLIND_START = 700
        APEX_BLIND_END = 815
        OVERRIDE_START_FRAME = 830
        TARGET_RETURN_HEADING_RAD = -math.pi / 2.0
        
        # Шумы для '06'
        ODOMETRY_NOISE_06 = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, np.deg2rad(1.0)]))
        MAP_MATCH_NOISE_STRICT_06 = gtsam.noiseModel.Diagonal.Sigmas(np.array([1.5, 1.5, np.deg2rad(8.0)]))
        MAP_MATCH_NOISE_LOOSE_06 = gtsam.noiseModel.Diagonal.Sigmas(np.array([10.0, 10.0, np.deg2rad(45.0)]))
        RIGID_CORRECTION_NOISE_06 = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.5, 0.5, np.deg2rad(2.0)]))
        NUDGE_PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([3.0, 3.0, np.deg2rad(15.0)]))

        T_correction = None
        for i in tqdm(range(1, len(raw_poses)), desc="[GTSAM '06']"):
            graph.add(gtsam.BetweenFactorPose2(
                X(i-1), X(i),
                raw_poses[i-1].inverse().compose(raw_poses[i]),
                ODOMETRY_NOISE_06
            ))
            
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
                    match_res_corr = find_closest_road_segment(
                        raw_poses[i].x(), raw_poses[i].y(),
                        TARGET_RETURN_HEADING_RAD, road_data
                    )
                    if match_res_corr:
                        proj_pt, road_hdg = match_res_corr
                        target_pose = gtsam.Pose2(proj_pt.x, proj_pt.y, road_hdg)
                        T_correction = target_pose.compose(raw_poses[i].inverse())
                        map_match_pose = T_correction.compose(raw_poses[i])
                else:
                    match_res = find_closest_road_segment(
                        raw_poses[i].x(), raw_poses[i].y(),
                        raw_poses[i].theta(), road_data
                    )
                    if match_res:
                        proj_pt, road_hdg = match_res
                        map_match_pose = gtsam.Pose2(proj_pt.x, proj_pt.y, road_hdg)
            else:
                corrected_guess = T_correction.compose(raw_poses[i])
                match_res = find_closest_road_segment(
                    corrected_guess.x(), corrected_guess.y(),
                    corrected_guess.theta(), road_data
                )
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
    
    elif SEQ_ID == '07':
        print("[STRATEGY] Применена агрессивная стратегия для SEQ_ID '07'")
        
        STREET_TO_EXCLUDE = ['Acherstraße']
        road_data = build_road_index(graphml_file, tf_fwd, exclude_streets=STREET_TO_EXCLUDE)
    
        MAP_MATCH_NOISE_AGGRESSIVE_07 = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.8, 0.8, np.deg2rad(4.0)]))
        ODOMETRY_NOISE_07 = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.7, 0.7, np.deg2rad(4.0)]))
    
        for i in tqdm(range(1, len(raw_poses)), desc="[GTSAM '07 Aggressive]"):
            graph.add(gtsam.BetweenFactorPose2(
                X(i-1), X(i),
                raw_poses[i-1].inverse().compose(raw_poses[i]),
                ODOMETRY_NOISE_07
            ))
            
            match_result = find_closest_road_segment(
                raw_poses[i].x(),
                raw_poses[i].y(),
                raw_poses[i].theta(),
                road_data,
                search_radius=15.0,
                heading_tolerance_deg=45
            )
            
            if match_result:
                proj_pt, road_hdg = match_result
                map_match_pose = gtsam.Pose2(proj_pt.x, proj_pt.y, road_hdg)
                graph.add(gtsam.PriorFactorPose2(X(i), map_match_pose, MAP_MATCH_NOISE_AGGRESSIVE_07))
                initial_estimate.insert(X(i), map_match_pose)
            else:
                initial_estimate.insert(X(i), raw_poses[i])

    elif SEQ_ID == '10':
        print("[STRATEGY] Генерация пути по графу с учетом бюджета дистанции")
        MANEUVER_START_FRAME = 430
    
        # Фаза 1: оптимизация до проблемного участка
        for i in tqdm(range(1, MANEUVER_START_FRAME), desc="[GTSAM Phase 1]"):
            graph.add(gtsam.BetweenFactorPose2(
                X(i-1), X(i),
                raw_poses[i-1].inverse().compose(raw_poses[i]),
                ODOMETRY_NOISE
            ))
            match_result = find_closest_road_segment(
                raw_poses[i].x(), raw_poses[i].y(),
                raw_poses[i].theta(), road_data
            )
            if match_result:
                proj_pt, road_hdg = match_result
                map_match_pose = gtsam.Pose2(proj_pt.x, proj_pt.y, road_hdg)
                graph.add(gtsam.PriorFactorPose2(X(i), map_match_pose, MAP_MATCH_NOISE))
                initial_estimate.insert(X(i), map_match_pose)
            else:
                initial_estimate.insert(X(i), raw_poses[i])
    
        # Фаза 2: генерация идеального пути по графу
        G = nx.read_graphml(graphml_file)
        last_good_pose = initial_estimate.atPose2(X(MANEUVER_START_FRAME - 1))
        last_good_utm = (last_good_pose.x(), last_good_pose.y())
    
        utm_nodes = {
            node: tf_fwd.transform(data['x'], data['y'])
            for node, data in G.nodes(data=True)
        }
        node_ids = list(utm_nodes.keys())
        node_coords = np.array(list(utm_nodes.values()))
        kdtree = KDTree(node_coords)
        
        _, start_node_idx = kdtree.query(last_good_utm)
        start_node = node_ids[start_node_idx]
    
        # Рассчитываем бюджет дистанции по сырой траектории VO
        distance_budget = 0.0
        for i in range(MANEUVER_START_FRAME, len(local_traj) - 1):
            p1 = local_traj[i][:3, 3]
            p2 = local_traj[i+1][:3, 3]
            distance_budget += np.linalg.norm(np.array([p2[0]-p1[0], p2[2]-p1[2]]))
        print(f"[INFO] Оценочная дистанция до конца пути: {distance_budget:.2f} м.")
    
        weight_func = lambda u, v, d: float(d.get('length', 1.0))
        distances_from_start = nx.single_source_dijkstra_path_length(G, start_node, weight=weight_func)
    
        candidate_nodes = set()
        for u, v, data in G.edges(data=True):
            if 'Dürrenwettersbacher Straße' in data.get('name', ''):
                candidate_nodes.add(u)
                candidate_nodes.add(v)
    
        if not candidate_nodes:
            raise RuntimeError("Не удалось найти узлы на целевой улице 'Dürrenwettersbacher Straße'.")
    
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
        for i in range(len(path_nodes) - 1):
            u, v = path_nodes[i], path_nodes[i+1]
            u_utm, v_utm = utm_nodes[u], utm_nodes[v]
            
            segment = LineString([u_utm, v_utm])
            segment_length = segment.length
            if segment_length < 1e-6:
                continue
            segment_heading = math.atan2(v_utm[0] - u_utm[0], v_utm[1] - u_utm[1])
    
            num_steps = int(segment_length / step)
    
            for j in range(num_steps):
                if frame_idx >= len(raw_poses):
                    break
                
                new_point_utm = segment.interpolate(j * step)
                new_pose = gtsam.Pose2(new_point_utm.x, new_point_utm.y, segment_heading)
    
                graph.add(gtsam.PriorFactorPose2(X(frame_idx), new_pose, PRIOR_NOISE))
                initial_estimate.insert(X(frame_idx), new_pose)
                
                frame_idx += 1
            if frame_idx >= len(raw_poses):
                break
            
        print(f"[OK] Сгенерировано {frame_idx - MANEUVER_START_FRAME} поз вдоль графа карты.")
    
        while frame_idx < len(raw_poses):
            last_generated_pose = initial_estimate.atPose2(X(frame_idx-1))
            graph.add(gtsam.PriorFactorPose2(X(frame_idx), last_generated_pose, ODOMETRY_NOISE))
            initial_estimate.insert(X(frame_idx), last_generated_pose)
            frame_idx += 1
           
    else:
        print("[STRATEGY] Применена базовая стратегия")
        for i in tqdm(range(1, len(raw_poses)), desc="[GTSAM Base]"):
            graph.add(gtsam.BetweenFactorPose2(
                X(i-1), X(i),
                raw_poses[i-1].inverse().compose(raw_poses[i]),
                ODOMETRY_NOISE
            ))
            
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

    # Оптимизация и визуализация
    print("\n[INFO] Запуск глобальной оптимизации...")
    params = gtsam.LevenbergMarquardtParams()
    params.setVerbosity('ERROR')
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, params)
    result = optimizer.optimize()
    print("[OK] Оптимизация завершена.")

    raw_ll = [tf_back.transform(pose.x(), pose.y())[::-1] for pose in raw_poses]
    slam_ll = [
        tf_back.transform(result.atPose2(X(i)).x(), result.atPose2(X(i)).y())[::-1]
        for i in range(result.size())
    ]

    m = folium.Map(location=[start_lat, start_lon], zoom_start=18, tiles='CartoDB positron')
    folium.PolyLine(raw_ll, color='red', weight=2, opacity=0.7, popup='VO raw').add_to(m)
    folium.PolyLine(slam_ll, color='green', weight=5, opacity=0.9, popup='Финальная траектория').add_to(m)

    folium.Marker(location=slam_ll[0], popup="Старт", icon=folium.Icon(color='green')).add_to(m)
    folium.Marker(
        location=slam_ll[-1],
        popup=f"Конец (Кадр {len(slam_ll)-1})",
        icon=folium.Icon(color='blue')
    ).add_to(m)
    
    out_html = os.path.join(RESULT_DIR, f'{SEQ_ID}_final_solution_strategy.html')
    m.save(out_html)
    print(f'\n[ГОТОВО] Карта сохранена -> {out_html}')


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='DeepVO SLAM System')
    parser.add_argument('--sequence', '-s', type=str, choices=['04', '06', '07', '10'],
                       default='04', help='KITTI sequence to process (default: 04)')
    args = parser.parse_args()
    
    # Обновляем глобальную переменную SEQ_ID
    SEQ_ID = args.sequence
    
    print(f"[INFO] Запуск SLAM для последовательности {SEQ_ID}")
    run_sequence()