#import mayavi.mlab as mlab
import numpy as np
import torch
import pickle
import os
import pdb
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl
from pyquaternion import Quaternion
from nuscenes.utils.geometry_utils import transform_matrix

box_colormap = [
    [1, 1, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 1, 0],
]

def visualize_points_and_boxes(dir_fig, pts, pred_boxes1, score1, gt_boxes1, 
                               pred_boxes2, score2, gt_boxes2, score_thresh):
    # Visualize Points, Boxes, Camera images
    # pts : [x, y, z, RCS, vx, vy, timestamp]
    # boxes : [x, y, z, dx, dy, dz, rot, velocity1, velocity2, class_idx]
    #! visual_pred True Not Implemented
    print('[SPALab]  Visualizing points and boxes. %s'%dir_fig)
    dir_data = '/mnt/mnt/sdd/ysshin/nuscenes/v1.0-trainval'
    #nusc_sample = nusc.get('sample', sample_token)

    fig = plt.figure(figsize=(50, 35))

    point_scale = 1
    ax1 = plt.subplot2grid(shape=(3,8), loc=(0,0), rowspan=3, colspan=4)
    plt.scatter(pts[:, 0], pts[:, 1])
    plt.plot(0, 0, 'x', color='red', markersize=20, linewidth=3)
    plt.xlim(-51.2, 51.2)
    plt.ylim(-51.2, 51.2)
    plt.grid()

    # Visualize Predicted Boxes1
    num_boxes = pred_boxes1.shape[0]
    for box_idx in range(num_boxes):
        if score1[box_idx] > score_thresh:
            box = patches.Rectangle((-pred_boxes1[box_idx, 3]/2,
                                     -pred_boxes1[box_idx, 4]/2),
                                     width=pred_boxes1[box_idx, 3],
                                     height=pred_boxes1[box_idx, 4],
                                     color='blue',
                                     facecolor='blue',
                                     alpha=0.3,
                                     linewidth=3)
            rot = mpl.transforms.Affine2D().rotate(pred_boxes1[box_idx, 6])
            trans = mpl.transforms.Affine2D().translate(tx=pred_boxes1[box_idx, 0],
                                                        ty=pred_boxes1[box_idx, 1])
            transform = rot + trans + ax1.transData
            box.set_transform(transform)
            ax1.add_patch(box)
            del box

    # Visualize GT Boxes1
    num_boxes = gt_boxes1.shape[0]
    for box_idx in range(num_boxes):
        box = patches.Rectangle((-gt_boxes1[box_idx, 3]/2,
                                 -gt_boxes1[box_idx, 4]/2),
                                 width=gt_boxes1[box_idx, 3],
                                 height=gt_boxes1[box_idx, 4],
                                 edgecolor='r',
                                 facecolor='r',
                                 alpha=0.3,
                                 linewidth=4)
        rot = mpl.transforms.Affine2D().rotate(gt_boxes1[box_idx, 6])
        trans = mpl.transforms.Affine2D().translate(tx=gt_boxes1[box_idx, 0],
                                                    ty=gt_boxes1[box_idx, 1])
        transform = rot + trans + ax1.transData
        box.set_transform(transform)
        ax1.add_patch(box)
        del box

    ax2 = plt.subplot2grid(shape=(3,8), loc=(0,4), rowspan=3, colspan=4)
    plt.scatter(pts[:, 0], pts[:, 1])
    plt.plot(0, 0, 'x', color='red', markersize=20, linewidth=3)
    plt.xlim(-51.2, 51.2)
    plt.ylim(-51.2, 51.2)
    plt.grid()

    # Visualize Predicted Boxes2
    num_boxes = pred_boxes2.shape[0]
    for box_idx in range(num_boxes):
        if score1[box_idx] > score_thresh:
            box = patches.Rectangle((-pred_boxes2[box_idx, 3]/2,
                                     -pred_boxes2[box_idx, 4]/2),
                                     width=pred_boxes2[box_idx, 3],
                                     height=pred_boxes2[box_idx, 4],
                                     color='blue',
                                     facecolor='blue',
                                     alpha=0.3,
                                     linewidth=3)
            rot = mpl.transforms.Affine2D().rotate(pred_boxes2[box_idx, 6])
            trans = mpl.transforms.Affine2D().translate(tx=pred_boxes2[box_idx, 0],
                                                        ty=pred_boxes2[box_idx, 1])
            transform = rot + trans + ax2.transData
            box.set_transform(transform)
            ax2.add_patch(box)
            del box

    # Visualize GT Boxes2
    num_boxes = gt_boxes2.shape[0]
    for box_idx in range(num_boxes):
        box = patches.Rectangle((-gt_boxes2[box_idx, 3]/2,
                                 -gt_boxes2[box_idx, 4]/2),
                                 width=gt_boxes2[box_idx, 3],
                                 height=gt_boxes2[box_idx, 4],
                                 edgecolor='r',
                                 facecolor='r',
                                 alpha=0.3,
                                 linewidth=4)
        rot = mpl.transforms.Affine2D().rotate(gt_boxes2[box_idx, 6])
        trans = mpl.transforms.Affine2D().translate(tx=gt_boxes2[box_idx, 0],
                                                    ty=gt_boxes2[box_idx, 1])
        transform = rot + trans + ax2.transData
        box.set_transform(transform)
        ax2.add_patch(box)
        del box

    plt.savefig(dir_fig)
    plt.close()

    #! Visual_pred Part
    #if bbox_gt.shape[0] != 0:
    #    for gt_idx in range(bbox_gt.shape[0]):
    #        box = patches.Rectangle((-bbox_gt[gt_idx, 3]/2,
    #                                 -bbox_gt[gt_idx, 4]/2),
    #                                width=bbox_gt[gt_idx, 3],
    #                                height=bbox_gt[gt_idx, 4],
    #                                color='red',
    #                                alpha=0.3,
    #                                linewidth=3)
    #        rot = mpl.transforms.Affine2D().rotate(bbox_gt[gt_idx, 5])
    #        trans = mpl.transforms.Affine2D().translate(tx=bbox_gt[gt_idx, 0],
    #                                                    ty=bbox_gt[gt_idx, 1])
    #        transform = rot + trans + ax1.transData
    #        box.set_transform(transform)
    #        ax1.add_patch(box)
    #        del box
    """

    token_cam_front = nusc_sample['data']['CAM_FRONT']
    token_cam_front_left = nusc_sample['data']['CAM_FRONT_LEFT']
    token_cam_front_right = nusc_sample['data']['CAM_FRONT_RIGHT']
    token_cam_back = nusc_sample['data']['CAM_BACK']
    token_cam_back_left = nusc_sample['data']['CAM_BACK_LEFT']
    token_cam_back_right = nusc_sample['data']['CAM_BACK_RIGHT']

    dir_img_front = nusc.get('sample_data', token_cam_front)['filename']
    dir_img_front = os.path.join(dir_data, dir_img_front)
    dir_img_front_left = nusc.get('sample_data', token_cam_front_left)['filename']
    dir_img_front_left = os.path.join(dir_data, dir_img_front_left)
    dir_img_front_right = nusc.get('sample_data', token_cam_front_right)['filename']
    dir_img_front_right = os.path.join(dir_data, dir_img_front_right)
    dir_img_back = nusc.get('sample_data', token_cam_back)['filename']
    dir_img_back = os.path.join(dir_data, dir_img_back)
    dir_img_back_left = nusc.get('sample_data', token_cam_back_left)['filename']
    dir_img_back_left = os.path.join(dir_data, dir_img_back_left)
    dir_img_back_right = nusc.get('sample_data', token_cam_back_right)['filename']
    dir_img_back_right = os.path.join(dir_data, dir_img_back_right)

    img_front = plt.imread(dir_img_front)
    img_front_left = plt.imread(dir_img_front_left)
    img_front_right = plt.imread(dir_img_front_right)
    img_back = plt.imread(dir_img_back)
    img_back_left = plt.imread(dir_img_back_left)
    img_back_right = plt.imread(dir_img_back_right)

    ax5 = plt.subplot2grid(shape=(3,8), loc=(0,6), rowspan=1, colspan=2)
    plt.imshow(img_front_left)
    plt.title('Front Left', fontsize=40)
    ax6 = plt.subplot2grid(shape=(3,8), loc=(1,6), rowspan=1, colspan=2)
    plt.imshow(img_front)
    plt.title('Front', fontsize=40)
    ax7 = plt.subplot2grid(shape=(3,8), loc=(2,6), rowspan=1, colspan=2)
    plt.imshow(img_front_right)
    plt.title('Front Right', fontsize=40)
    ax2 = plt.subplot2grid(shape=(3,8), loc=(0, 0), rowspan=1, colspan=2)
    plt.imshow(img_back_left)
    plt.title('Back Left', fontsize=40)
    ax3 = plt.subplot2grid(shape=(3,8), loc=(1, 0), rowspan=1, colspan=2)
    plt.imshow(img_back)
    plt.title('BACK', fontsize=40)
    ax4 = plt.subplot2grid(shape=(3,8), loc=(2, 0), rowspan=1, colspan=2)
    plt.imshow(img_back_right)
    plt.title('Back Right', fontsize=40)
    """
    
def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False


def rotate_points_along_z(points, angle):
    """
    Args:
        points: (B, N, 3 + C)
        angle: (B), angle along z-axis, angle increases x ==> y
    Returns:

    """
    points, is_numpy = check_numpy_to_torch(points)
    angle, _ = check_numpy_to_torch(angle)

    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    zeros = angle.new_zeros(points.shape[0])
    ones = angle.new_ones(points.shape[0])
    rot_matrix = torch.stack((
        cosa,  sina, zeros,
        -sina, cosa, zeros,
        zeros, zeros, ones
    ), dim=1).view(-1, 3, 3).float()
    points_rot = torch.matmul(points[:, :, 0:3], rot_matrix)
    points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)
    return points_rot.numpy() if is_numpy else points_rot


def boxes_to_corners_3d(boxes3d):
    """
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
    Args:
        boxes3d:  (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center

    Returns:
    """
    boxes3d, is_numpy = check_numpy_to_torch(boxes3d)

    template = boxes3d.new_tensor((
        [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
        [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1],
    )) / 2

    corners3d = boxes3d[:, None, 3:6].repeat(1, 8, 1) * template[None, :, :]
    corners3d = rotate_points_along_z(corners3d.view(-1, 8, 3), boxes3d[:, 6]).view(-1, 8, 3)
    corners3d += boxes3d[:, None, 0:3]

    return corners3d.numpy() if is_numpy else corners3d


def visualize_pts(pts, fig=None, bgcolor=(0, 0, 0), fgcolor=(1.0, 1.0, 1.0),
                  show_intensity=False, size=(600, 600), draw_origin=True):
    if not isinstance(pts, np.ndarray):
        pts = pts.cpu().numpy()
    if fig is None:
        fig = mlab.figure(figure=None, bgcolor=bgcolor, fgcolor=fgcolor, engine=None, size=size)

    if show_intensity:
        G = mlab.points3d(pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3], mode='point',
                          colormap='gnuplot', scale_factor=1, figure=fig)
    else:
        G = mlab.points3d(pts[:, 0], pts[:, 1], pts[:, 2], mode='point',
                          colormap='gnuplot', scale_factor=1, figure=fig)
    if draw_origin:
        mlab.points3d(0, 0, 0, color=(1, 1, 1), mode='cube', scale_factor=0.2)
        mlab.plot3d([0, 3], [0, 0], [0, 0], color=(0, 0, 1), tube_radius=0.1)
        mlab.plot3d([0, 0], [0, 3], [0, 0], color=(0, 1, 0), tube_radius=0.1)
        mlab.plot3d([0, 0], [0, 0], [0, 3], color=(1, 0, 0), tube_radius=0.1)

    return fig


def draw_sphere_pts(pts, color=(0, 1, 0), fig=None, bgcolor=(0, 0, 0), scale_factor=0.2):
    if not isinstance(pts, np.ndarray):
        pts = pts.cpu().numpy()

    if fig is None:
        fig = mlab.figure(figure=None, bgcolor=bgcolor, fgcolor=None, engine=None, size=(600, 600))

    if isinstance(color, np.ndarray) and color.shape[0] == 1:
        color = color[0]
        color = (color[0] / 255.0, color[1] / 255.0, color[2] / 255.0)

    if isinstance(color, np.ndarray):
        pts_color = np.zeros((pts.__len__(), 4), dtype=np.uint8)
        pts_color[:, 0:3] = color
        pts_color[:, 3] = 255
        G = mlab.points3d(pts[:, 0], pts[:, 1], pts[:, 2], np.arange(0, pts_color.__len__()), mode='sphere',
                          scale_factor=scale_factor, figure=fig)
        G.glyph.color_mode = 'color_by_scalar'
        G.glyph.scale_mode = 'scale_by_vector'
        G.module_manager.scalar_lut_manager.lut.table = pts_color
    else:
        mlab.points3d(pts[:, 0], pts[:, 1], pts[:, 2], mode='sphere', color=color,
                      colormap='gnuplot', scale_factor=scale_factor, figure=fig)

    mlab.points3d(0, 0, 0, color=(1, 1, 1), mode='cube', scale_factor=0.2)
    mlab.plot3d([0, 3], [0, 0], [0, 0], color=(0, 0, 1), line_width=3, tube_radius=None, figure=fig)
    mlab.plot3d([0, 0], [0, 3], [0, 0], color=(0, 1, 0), line_width=3, tube_radius=None, figure=fig)
    mlab.plot3d([0, 0], [0, 0], [0, 3], color=(1, 0, 0), line_width=3, tube_radius=None, figure=fig)

    return fig


def draw_grid(x1, y1, x2, y2, fig, tube_radius=None, color=(0.5, 0.5, 0.5)):
    mlab.plot3d([x1, x1], [y1, y2], [0, 0], color=color, tube_radius=tube_radius, line_width=1, figure=fig)
    mlab.plot3d([x2, x2], [y1, y2], [0, 0], color=color, tube_radius=tube_radius, line_width=1, figure=fig)
    mlab.plot3d([x1, x2], [y1, y1], [0, 0], color=color, tube_radius=tube_radius, line_width=1, figure=fig)
    mlab.plot3d([x1, x2], [y2, y2], [0, 0], color=color, tube_radius=tube_radius, line_width=1, figure=fig)
    return fig


def draw_multi_grid_range(fig, grid_size=20, bv_range=(-60, -60, 60, 60)):
    for x in range(bv_range[0], bv_range[2], grid_size):
        for y in range(bv_range[1], bv_range[3], grid_size):
            fig = draw_grid(x, y, x + grid_size, y + grid_size, fig)

    return fig


def draw_scenes(points, gt_boxes=None, ref_boxes=None, ref_scores=None, ref_labels=None):
    if not isinstance(points, np.ndarray):
        points = points.cpu().numpy()
    if ref_boxes is not None and not isinstance(ref_boxes, np.ndarray):
        ref_boxes = ref_boxes.cpu().numpy()
    if gt_boxes is not None and not isinstance(gt_boxes, np.ndarray):
        gt_boxes = gt_boxes.cpu().numpy()
    if ref_scores is not None and not isinstance(ref_scores, np.ndarray):
        ref_scores = ref_scores.cpu().numpy()
    if ref_labels is not None and not isinstance(ref_labels, np.ndarray):
        ref_labels = ref_labels.cpu().numpy()

    fig = visualize_pts(points)
    fig = draw_multi_grid_range(fig, bv_range=(0, -40, 80, 40))
    if gt_boxes is not None:
        corners3d = boxes_to_corners_3d(gt_boxes)
        fig = draw_corners3d(corners3d, fig=fig, color=(0, 0, 1), max_num=100)

    if ref_boxes is not None and len(ref_boxes) > 0:
        ref_corners3d = boxes_to_corners_3d(ref_boxes)
        if ref_labels is None:
            fig = draw_corners3d(ref_corners3d, fig=fig, color=(0, 1, 0), cls=ref_scores, max_num=100)
        else:
            for k in range(ref_labels.min(), ref_labels.max() + 1):
                cur_color = tuple(box_colormap[k % len(box_colormap)])
                mask = (ref_labels == k)
                fig = draw_corners3d(ref_corners3d[mask], fig=fig, color=cur_color, cls=ref_scores[mask], max_num=100)
    mlab.view(azimuth=-179, elevation=54.0, distance=104.0, roll=90.0)
    return fig


def draw_corners3d(corners3d, fig, color=(1, 1, 1), line_width=2, cls=None, tag='', max_num=500, tube_radius=None):
    """
    :param corners3d: (N, 8, 3)
    :param fig:
    :param color:
    :param line_width:
    :param cls:
    :param tag:
    :param max_num:
    :return:
    """
    import mayavi.mlab as mlab
    num = min(max_num, len(corners3d))
    for n in range(num):
        b = corners3d[n]  # (8, 3)

        if cls is not None:
            if isinstance(cls, np.ndarray):
                mlab.text3d(b[6, 0], b[6, 1], b[6, 2], '%.2f' % cls[n], scale=(0.3, 0.3, 0.3), color=color, figure=fig)
            else:
                mlab.text3d(b[6, 0], b[6, 1], b[6, 2], '%s' % cls[n], scale=(0.3, 0.3, 0.3), color=color, figure=fig)

        for k in range(0, 4):
            i, j = k, (k + 1) % 4
            mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]], color=color, tube_radius=tube_radius,
                        line_width=line_width, figure=fig)

            i, j = k + 4, (k + 1) % 4 + 4
            mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]], color=color, tube_radius=tube_radius,
                        line_width=line_width, figure=fig)

            i, j = k, k + 4
            mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]], color=color, tube_radius=tube_radius,
                        line_width=line_width, figure=fig)

        i, j = 0, 5
        mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]], color=color, tube_radius=tube_radius,
                    line_width=line_width, figure=fig)
        i, j = 1, 4
        mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]], color=color, tube_radius=tube_radius,
                    line_width=line_width, figure=fig)

    return fig
