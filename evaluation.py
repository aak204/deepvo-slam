# /content/DeepVO/evaluation.py (ФИНАЛЬНАЯ ВЕРСИЯ)
#
# Copyright Qing Li (hello.qingli@gmail.com) 2018. All Rights Reserved.
#
# References: 1. KITTI odometry development kit: http://www.cvlibs.net/datasets/kitti/eval_odometry.php
#             2. A Geiger, P Lenz, R Urtasun. Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite. CVPR 2012.
#

import glob
import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

plt.switch_backend('agg')
import matplotlib.backends.backend_pdf
# Убедись, что папка tools есть в sys.path, если скрипт запускается из корня
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import tools.transformations as tr
from tools.pose_evaluation_utils import quat_pose_to_mat
from params_test import par

# ==============================================================================
# === КЛАСС ОЦЕНКИ KITTI (взят из твоего файла, без изменений) ===
# ==============================================================================
class kittiOdomEval():
    # ... СКОПИРУЙ СЮДА ВЕСЬ КОД КЛАССА kittiOdomEval ИЗ СТАРОГО ФАЙЛА ...
    # ... Я НЕ БУДУ ВСТАВЛЯТЬ ЕГО ЗДЕСЬ, ЧТОБЫ НЕ УДЛИНЯТЬ ОТВЕТ ...
    # ... Он остается без изменений ...
    def __init__(self, config):
        # Теперь config.gt_dir будет взят из params_test.py
        assert os.path.exists(config.gt_dir), "Error of ground_truth pose path!"
        gt_files = glob.glob(config.gt_dir + '/*.txt')
        gt_files = [os.path.split(f)[1] for f in gt_files]
        self.seqs_with_gt = [os.path.splitext(f)[0] for f in gt_files]

        self.lengths = [100,200,300,400,500,600,700,800]
        self.num_lengths = len(self.lengths)
        self.gt_dir     = config.gt_dir
        self.result_dir = config.result_dir
        self.eval_seqs  = []
        
        # evalute all files in the folder
        if config.eva_seqs == '*':
            if not os.path.exists(self.result_dir):
                print(f'File path error! Directory not found: {self.result_dir}')
                exit()
            if os.path.exists(self.result_dir + '/all_stats.txt'): 
                os.remove(self.result_dir + '/all_stats.txt')
            files = glob.glob(self.result_dir + '/*.txt')
            assert files, "There is no trajectory files in: {}".format(self.result_dir)
            for f in files:
                dirname, basename = os.path.split(f)
                file_name = os.path.splitext(basename)[0]
                self.eval_seqs.append(str(file_name))
        else:
            seqs = config.eva_seqs.split(',')
            self.eval_seqs = [str(s) for s in seqs]

    def toCameraCoord(self, pose_mat):
        '''
            Convert the pose of lidar coordinate to camera coordinate
        '''
        R_C2L = np.array([[0,   0,   1,  0],
                          [-1,  0,   0,  0],
                          [0,  -1,   0,  0],
                          [0,   0,   0,  1]])
        inv_R_C2L = np.linalg.inv(R_C2L)            
        R = np.dot(inv_R_C2L, pose_mat)
        rot = np.dot(R, R_C2L)
        return rot 

    def loadPoses(self, file_name, toCameraCoord):
        '''
            Each line in the file should follow one of the following structures
            (1) idx pose(3x4 matrix in terms of 12 numbers)
            (2) pose(3x4 matrix in terms of 12 numbers)
        '''
        f = open(file_name, 'r')
        s = f.readlines()
        f.close()
        file_len = len(s)
        poses = {}
        frame_idx = 0
        for cnt, line in enumerate(s):
            P = np.eye(4)
            line_split = [float(i) for i in line.split()]
            withIdx = int(len(line_split)==13)
            for row in range(3):
                for col in range(4):
                    P[row, col] = line_split[row*4 + col + withIdx]
            if withIdx:
                frame_idx = line_split[0]
            else:
                frame_idx = cnt
            if toCameraCoord:
                poses[frame_idx] = self.toCameraCoord(P)
            else:
                poses[frame_idx] = P
        return poses

    def trajectoryDistances(self, poses):
        '''
            Compute the length of the trajectory
            poses dictionary: [frame_idx: pose]
        '''
        dist = [0]
        sort_frame_idx = sorted(poses.keys())
        for i in range(len(sort_frame_idx)-1):
            cur_frame_idx = sort_frame_idx[i]
            next_frame_idx = sort_frame_idx[i+1]
            P1 = poses[cur_frame_idx]
            P2 = poses[next_frame_idx]
            dx = P1[0,3] - P2[0,3]
            dy = P1[1,3] - P2[1,3]
            dz = P1[2,3] - P2[2,3]
            dist.append(dist[i]+np.sqrt(dx**2+dy**2+dz**2))    
        self.distance = dist[-1]
        return dist

    def rotationError(self, pose_error):
        a = pose_error[0,0]
        b = pose_error[1,1]
        c = pose_error[2,2]
        d = 0.5*(a+b+c-1.0)
        return np.arccos(max(min(d,1.0),-1.0))

    def translationError(self, pose_error):
        dx = pose_error[0,3]
        dy = pose_error[1,3]
        dz = pose_error[2,3]
        return np.sqrt(dx**2+dy**2+dz**2)

    def lastFrameFromSegmentLength(self, dist, first_frame, len_):
        for i in range(first_frame, len(dist), 1):
            if dist[i] > (dist[first_frame] + len_):
                return i
        return -1

    def calcSequenceErrors(self, poses_gt, poses_result):
        err = []
        self.max_speed = 0
        dist = self.trajectoryDistances(poses_gt)
        self.step_size = 10
        for first_frame in range(0, len(poses_gt), self.step_size):
            for i in range(self.num_lengths):
                len_ = self.lengths[i]
                last_frame = self.lastFrameFromSegmentLength(dist, first_frame, len_)
                if last_frame == -1 or not(last_frame in poses_result.keys()) or not(first_frame in poses_result.keys()):
                    continue
                pose_delta_gt = np.dot(np.linalg.inv(poses_gt[first_frame]), poses_gt[last_frame])
                pose_delta_result = np.dot(np.linalg.inv(poses_result[first_frame]), poses_result[last_frame])
                pose_error = np.dot(np.linalg.inv(pose_delta_result), pose_delta_gt)
                r_err = self.rotationError(pose_error)
                t_err = self.translationError(pose_error)
                num_frames = last_frame - first_frame + 1.0
                speed = len_ / (0.1*num_frames)
                if speed > self.max_speed:
                    self.max_speed = speed
                err.append([first_frame, r_err/len_, t_err/len_, len_, speed])
        return err
        
    def saveSequenceErrors(self, err, file_name):
        fp = open(file_name,'w')
        for i in err:
            line_to_write = " ".join([str(j) for j in i])
            fp.writelines(line_to_write+"\n")
        fp.close()

    def computeOverallErr(self, seq_err):
        t_err = 0
        r_err = 0
        if not seq_err:
            return 0, 0
        seq_len = len(seq_err)
        for item in seq_err:
            r_err += item[1]
            t_err += item[2]
        ave_t_err = t_err / seq_len
        ave_r_err = r_err / seq_len
        return ave_t_err, ave_r_err 

    def plot_xyz(self, seq, poses_ref, poses_pred, plot_path_dir):
        
        def traj_xyz(axarr, positions_xyz, style='-', color='black', title="", label="", alpha=1.0):
            x = range(0, len(positions_xyz))
            xlabel = "index"
            ylabels = ["$x$ (m)", "$y$ (m)", "$z$ (m)"]
            for i in range(0, 3):
                axarr[i].plot(x, positions_xyz[:, i], style, color=color, label=label, alpha=alpha)
                axarr[i].set_ylabel(ylabels[i])
                axarr[i].legend(loc="upper right", frameon=True)
            axarr[2].set_xlabel(xlabel)
            if title:
                axarr[0].set_title('XYZ')           

        fig, axarr = plt.subplots(3, sharex="col", figsize=tuple([20, 10]))  
        pred_xyz = np.array([p[:3, 3] for _,p in poses_pred.items()])
        traj_xyz(axarr, pred_xyz, '-', 'b', title='XYZ', label='Ours', alpha=1.0)
        if poses_ref:
            ref_xyz = np.array([p[:3, 3] for _,p in poses_ref.items()])
            traj_xyz(axarr, ref_xyz, '-', 'r', label='GT', alpha=1.0)
      
        name = "{}_xyz".format(seq)
        plt.savefig(plot_path_dir +  "/" + name + ".png", bbox_inches='tight', pad_inches=0.1)
        pdf = matplotlib.backends.backend_pdf.PdfPages(plot_path_dir +  "/" + name + ".pdf")        
        fig.tight_layout()
        pdf.savefig(fig)       
        pdf.close()

    def plot_rpy(self, seq, poses_ref, poses_pred, plot_path_dir, axes='szxy'):
        
        def traj_rpy(axarr, orientations_euler, style='-', color='black', title="", label="", alpha=1.0):
            x = range(0, len(orientations_euler))
            xlabel = "index"
            ylabels = ["$roll$ (deg)", "$pitch$ (deg)", "$yaw$ (deg)"]
            for i in range(0, 3):
                axarr[i].plot(x, np.rad2deg(orientations_euler[:, i]), style,
                            color=color, label=label, alpha=alpha)
                axarr[i].set_ylabel(ylabels[i])
                axarr[i].legend(loc="upper right", frameon=True)
            axarr[2].set_xlabel(xlabel)
            if title:
                axarr[0].set_title('PRY')           

        fig_rpy, axarr_rpy = plt.subplots(3, sharex="col", figsize=tuple([20, 10]))
        pred_rpy = np.array([tr.euler_from_matrix(p, axes=axes) for _,p in poses_pred.items()])
        traj_rpy(axarr_rpy, pred_rpy, '-', 'b', title='RPY', label='Ours', alpha=1.0)
        if poses_ref:
            ref_rpy = np.array([tr.euler_from_matrix(p, axes=axes) for _,p in poses_ref.items()])
            traj_rpy(axarr_rpy, ref_rpy, '-', 'r', label='GT', alpha=1.0)

        name = "{}_rpy".format(seq)
        plt.savefig(plot_path_dir +  "/" + name + ".png", bbox_inches='tight', pad_inches=0.1)
        pdf = matplotlib.backends.backend_pdf.PdfPages(plot_path_dir +  "/" + name + ".pdf")        
        fig_rpy.tight_layout()
        pdf.savefig(fig_rpy)       
        pdf.close()

    def plotPath_2D_3(self, seq, poses_gt, poses_result, plot_path_dir):
        fontsize_ = 10
        plot_keys = ["Ground Truth", "Ours"]
        start_point = [0, 0]
        style_pred = 'b-'
        style_gt = 'r-'
        style_O = 'ko'
        if poses_gt: 
            poses_gt = [(k,poses_gt[k]) for k in sorted(poses_gt.keys())]
            x_gt = np.asarray([pose[0,3] for _,pose in poses_gt])
            y_gt = np.asarray([pose[1,3] for _,pose in poses_gt])
            z_gt = np.asarray([pose[2,3] for _,pose in poses_gt])
        poses_result = [(k,poses_result[k]) for k in sorted(poses_result.keys())]
        x_pred = np.asarray([pose[0,3] for _,pose in poses_result])
        y_pred = np.asarray([pose[1,3] for _,pose in poses_result])
        z_pred = np.asarray([pose[2,3] for _,pose in poses_result])
        
        fig = plt.figure(figsize=(20,6), dpi=100)
        plt.subplot(1,3,1)
        ax = plt.gca()
        if poses_gt: plt.plot(x_gt, z_gt, style_gt, label=plot_keys[0])
        plt.plot(x_pred, z_pred, style_pred, label=plot_keys[1])
        plt.plot(start_point[0], start_point[1], style_O, label='Start Point')
        plt.legend(loc="upper right", prop={'size':fontsize_})
        plt.xlabel('x (m)', fontsize=fontsize_)
        plt.ylabel('z (m)', fontsize=fontsize_)
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        xmean = np.mean(xlim)
        ymean = np.mean(ylim)
        plot_radius = max([abs(lim - mean_) for lims, mean_ in ((xlim, xmean), (ylim, ymean)) for lim in lims])
        ax.set_xlim([xmean - plot_radius, xmean + plot_radius])
        ax.set_ylim([ymean - plot_radius, ymean + plot_radius])

        plt.subplot(1,3,2)
        ax = plt.gca()
        if poses_gt: plt.plot(x_gt, y_gt, style_gt, label=plot_keys[0])
        plt.plot(x_pred, y_pred, style_pred, label=plot_keys[1])
        plt.plot(start_point[0], start_point[1], style_O, label='Start Point')
        plt.legend(loc="upper right", prop={'size':fontsize_})
        plt.xlabel('x (m)', fontsize=fontsize_)
        plt.ylabel('y (m)', fontsize=fontsize_)
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        xmean = np.mean(xlim)
        ymean = np.mean(ylim)
        ax.set_xlim([xmean - plot_radius, xmean + plot_radius])
        ax.set_ylim([ymean - plot_radius, ymean + plot_radius])

        plt.subplot(1,3,3)
        ax = plt.gca()
        if poses_gt: plt.plot(y_gt, z_gt, style_gt, label=plot_keys[0])
        plt.plot(y_pred, z_pred, style_pred, label=plot_keys[1])
        plt.plot(start_point[0], start_point[1], style_O, label='Start Point')
        plt.legend(loc="upper right", prop={'size':fontsize_})
        plt.xlabel('y (m)', fontsize=fontsize_)
        plt.ylabel('z (m)', fontsize=fontsize_)
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        xmean = np.mean(xlim)
        ymean = np.mean(ylim)
        ax.set_xlim([xmean - plot_radius, xmean + plot_radius])
        ax.set_ylim([ymean - plot_radius, ymean + plot_radius])

        png_title = "{}_path".format(seq)
        plt.savefig(plot_path_dir +  "/" + png_title + ".png", bbox_inches='tight', pad_inches=0.1)
        pdf = matplotlib.backends.backend_pdf.PdfPages(plot_path_dir +  "/" + png_title + ".pdf")        
        fig.tight_layout()
        pdf.savefig(fig)  
        plt.close()

    def plotPath_3D(self, seq, poses_gt, poses_result, plot_path_dir):
        from mpl_toolkits.mplot3d import Axes3D
        start_point = [[0], [0], [0]]
        fontsize_ = 8
        style_pred = 'b-'
        style_gt = 'r-'
        style_O = 'ko'
        poses_dict = {}      
        poses_dict["Ours"] = poses_result
        if poses_gt:
            poses_dict["Ground Truth"] = poses_gt
        fig = plt.figure(figsize=(8,8), dpi=110)
        ax = fig.add_subplot(111, projection='3d')
        for key,_ in poses_dict.items():
            plane_point = []
            for frame_idx in sorted(poses_dict[key].keys()):
                pose = poses_dict[key][frame_idx]
                plane_point.append([pose[0,3], pose[2,3], pose[1,3]])
            plane_point = np.asarray(plane_point)
            style = style_pred if key == 'Ours' else style_gt
            plt.plot(plane_point[:,0], plane_point[:,1], plane_point[:,2], style, label=key)  
        plt.plot(start_point[0], start_point[1], start_point[2], style_O, label='Start Point')
        xlim = ax.get_xlim3d()
        ylim = ax.get_ylim3d()
        zlim = ax.get_zlim3d()
        xmean = np.mean(xlim)
        ymean = np.mean(ylim)
        zmean = np.mean(zlim)
        plot_radius = max([abs(lim - mean_) for lims, mean_ in ((xlim, xmean), (ylim, ymean), (zlim, zmean)) for lim in lims])
        ax.set_xlim3d([xmean - plot_radius, xmean + plot_radius])
        ax.set_ylim3d([ymean - plot_radius, ymean + plot_radius])
        ax.set_zlim3d([zmean - plot_radius, zmean + plot_radius])
        ax.legend()
        ax.set_xlabel('x (m)', fontsize=fontsize_)
        ax.set_ylabel('z (m)', fontsize=fontsize_)
        ax.set_zlabel('y (m)', fontsize=fontsize_)
        ax.view_init(elev=20., azim=-35)

        png_title = "{}_path_3D".format(seq)
        plt.savefig(plot_path_dir+"/"+png_title+".png", bbox_inches='tight', pad_inches=0.1)
        pdf = matplotlib.backends.backend_pdf.PdfPages(plot_path_dir +  "/" + png_title + ".pdf")        
        fig.tight_layout()
        pdf.savefig(fig)  
        plt.close()

    def plotError_segment(self, seq, avg_segment_errs, plot_error_dir):
        fontsize_ = 15
        plot_y_t, plot_y_r, plot_x = [], [], []
        for idx, value in avg_segment_errs.items():
            if not value: continue
            plot_x.append(idx)
            plot_y_t.append(value[0] * 100)
            plot_y_r.append(value[1]/np.pi * 180)
        
        fig = plt.figure(figsize=(15,6), dpi=100)
        plt.subplot(1,2,1)
        plt.plot(plot_x, plot_y_t, 'ks-')
        if plot_x: plt.axis([100, np.max(plot_x), 0, np.max(plot_y_t)*(1+0.1)])
        plt.xlabel('Path Length (m)',fontsize=fontsize_)
        plt.ylabel('Translation Error (%)',fontsize=fontsize_)

        plt.subplot(1,2,2)
        plt.plot(plot_x, plot_y_r, 'ks-')
        if plot_x: plt.axis([100, np.max(plot_x), 0, np.max(plot_y_r)*(1+0.1)])
        plt.xlabel('Path Length (m)',fontsize=fontsize_)
        plt.ylabel('Rotation Error (deg/m)',fontsize=fontsize_)
        png_title = "{}_error_seg".format(seq)
        plt.savefig(plot_error_dir +  "/" + png_title + ".png", bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)

    def plotError_speed(self, seq, avg_speed_errs, plot_error_dir):
        fontsize_ = 15
        plot_y_t, plot_y_r, plot_x = [], [], []
        for idx, value in avg_speed_errs.items():
            if not value: continue
            plot_x.append(idx * 3.6)
            plot_y_t.append(value[0] * 100)
            plot_y_r.append(value[1]/np.pi * 180)
        
        fig = plt.figure(figsize=(15,6), dpi=100)
        plt.subplot(1,2,1)        
        plt.plot(plot_x, plot_y_t, 'ks-')
        if plot_x: plt.axis([np.min(plot_x), np.max(plot_x), 0, np.max(plot_y_t)*(1+0.1)])
        plt.xlabel('Speed (km/h)',fontsize = fontsize_)
        plt.ylabel('Translation Error (%)',fontsize = fontsize_)

        plt.subplot(1,2,2)
        plt.plot(plot_x, plot_y_r, 'ks-')
        if plot_x: plt.axis([np.min(plot_x), np.max(plot_x), 0, np.max(plot_y_r)*(1+0.1)])
        plt.xlabel('Speed (km/h)',fontsize = fontsize_)
        plt.ylabel('Rotation Error (deg/m)',fontsize = fontsize_)
        png_title = "{}_error_speed".format(seq)
        plt.savefig(plot_error_dir +  "/" + png_title + ".png", bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)

    def computeSegmentErr(self, seq_errs):
        segment_errs = {len_: [] for len_ in self.lengths}
        avg_segment_errs = {}
        for err in seq_errs:
            segment_errs[err[3]].append([err[2], err[1]])
        for len_ in self.lengths:
            if segment_errs[len_]:
                avg_t_err = np.mean(np.asarray(segment_errs[len_])[:,0])
                avg_r_err = np.mean(np.asarray(segment_errs[len_])[:,1])
                avg_segment_errs[len_] = [avg_t_err, avg_r_err]
            else:
                avg_segment_errs[len_] = []
        return avg_segment_errs

    def computeSpeedErr(self, seq_errs):
        segment_errs = {s: [] for s in range(2, 25, 2)}
        avg_segment_errs = {}
        for err in seq_errs:
            speed, t_err, r_err = err[4], err[2], err[1]
            for key in segment_errs.keys():
                if np.abs(speed - key) < 2.0:
                    segment_errs[key].append([t_err, r_err])
        for key in segment_errs.keys():
            if segment_errs[key]:
                avg_t_err = np.mean(np.asarray(segment_errs[key])[:,0])
                avg_r_err = np.mean(np.asarray(segment_errs[key])[:,1])
                avg_segment_errs[key] = [avg_t_err, avg_r_err]
            else:
                avg_segment_errs[key] = []
        return avg_segment_errs

    def eval(self, toCameraCoord):
        eval_dir = self.result_dir
        if not os.path.exists(eval_dir): os.makedirs(eval_dir)        
        total_err = []
        ave_errs = {}       
        for seq in self.eval_seqs:
            eva_seq_dir = os.path.join(eval_dir, '{}_eval'.format(seq))
            pred_file_name = self.result_dir + '/{}.txt'.format(seq)
            gt_file_name   = self.gt_dir + '/{}.txt'.format(seq)
            assert os.path.exists(pred_file_name), "File path error: {}".format(pred_file_name)
            poses_result = self.loadPoses(pred_file_name, toCameraCoord=toCameraCoord)
            if not os.path.exists(eva_seq_dir): os.makedirs(eva_seq_dir) 
            if seq not in self.seqs_with_gt:
                print(f"Ground truth for sequence {seq} not found. Skipping evaluation, only plotting trajectory.")
                self.calcSequenceErrors(poses_result, poses_result)
                print(f"\nSequence: {seq}")
                print(f'Distance (m): {self.distance:.2f}')
                print(f'Max speed (km/h): {self.max_speed*3.6:.2f}')
                self.plot_rpy(seq, None, poses_result, eva_seq_dir)
                self.plot_xyz(seq, None, poses_result, eva_seq_dir)
                self.plotPath_3D(seq, None, poses_result, eva_seq_dir)
                self.plotPath_2D_3(seq, None, poses_result, eva_seq_dir)
                continue
            poses_gt = self.loadPoses(gt_file_name, toCameraCoord=False)
            seq_err = self.calcSequenceErrors(poses_gt, poses_result)
            self.saveSequenceErrors(seq_err, eva_seq_dir + '/{}_error.txt'.format(seq))
            total_err += seq_err
            avg_segment_errs = self.computeSegmentErr(seq_err)
            avg_speed_errs   = self.computeSpeedErr(seq_err)
            ave_t_err, ave_r_err = self.computeOverallErr(seq_err)
            print ("\nSequence: " + str(seq))
            print (f'Distance (m): {self.distance:.2f}')
            print (f'Max speed (km/h): {self.max_speed*3.6:.2f}')
            print ("Average sequence translational RMSE (%):   {0:.4f}".format(ave_t_err * 100))
            print ("Average sequence rotational error (deg/m): {0:.4f}\n".format(ave_r_err/np.pi * 180))
            with open(eva_seq_dir + '/%s_stats.txt' % seq, 'w') as f:
                f.writelines('Average sequence translation RMSE (%):    {0:.4f}\n'.format(ave_t_err * 100))
                f.writelines('Average sequence rotation error (deg/m):  {0:.4f}'.format(ave_r_err/np.pi * 180))
            ave_errs[seq] = [ave_t_err, ave_r_err]
            self.plot_rpy(seq, poses_gt, poses_result, eva_seq_dir)
            self.plot_xyz(seq, poses_gt, poses_result, eva_seq_dir)
            self.plotPath_3D(seq, poses_gt, poses_result, eva_seq_dir)
            self.plotPath_2D_3(seq, poses_gt, poses_result, eva_seq_dir)
            self.plotError_segment(seq, avg_segment_errs, eva_seq_dir)
            self.plotError_speed(seq, avg_speed_errs, eva_seq_dir)
            plt.close('all')

# ==============================================================================
# === ТОЧКА ВХОДА ===
# ==============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='KITTI Evaluation toolkit')
    parser.add_argument('--mode',       type=str, required=True, choices=['raw', 'filtered'], help='Режим оценки: "raw" для сырых данных, "filtered" для отфильтрованных.')
    parser.add_argument('--eva_seqs',   type=str, default=','.join(par.test_video), help='Последовательности для оценки (через запятую).')
    
    args = parser.parse_args()

    # --- Динамически определяем папку с результатами ---
    # par.result_dir - это базовая папка, например /content/DeepVO/_SAVE_RESULTS/kitti_original_params
    # Мы добавляем к ней /raw или /filtered
    result_dir_for_mode = os.path.join(par.result_dir, args.mode)

    print(f"===== РЕЖИМ ОЦЕНКИ: {args.mode.upper()} =====")
    print(f"Папка с Ground Truth: {par.pose_dir}")
    print(f"Папка с результатами: {result_dir_for_mode}")

    # --- Создаем новый объект config для класса оценки ---
    eval_config = argparse.Namespace(
        gt_dir=par.pose_dir,
        result_dir=result_dir_for_mode,
        eva_seqs=args.eva_seqs
    )
    
    pose_eval = kittiOdomEval(eval_config)
    pose_eval.eval(toCameraCoord=False) # В нашем test.py мы уже работаем в координатах камеры