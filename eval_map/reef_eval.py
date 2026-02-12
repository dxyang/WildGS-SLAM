import math
import os
from typing import Tuple, List
import cv2
import json
import yaml
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

import viser
import nerfview
from scipy.spatial.transform import Rotation as R
from gsplat.rendering import rasterization

from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from thirdparty.gaussian_splatting.gaussian_renderer import render
from thirdparty.gaussian_splatting.scene.gaussian_model import GaussianModel
from eval_map.utils import *

from slam_package import EVOEvaluator

device = torch.device("cuda")


K_TEKTITE_DELAY_S = 0.2906 # do RAW_IMAGE_TIMESTAMP - K_TEKTITE_DELAY or odom_timestamp + K_TEKTITE_DELAY
K_YAWZI_DELAY_S = 0.04152 # do RAW_IMAGE_TIMESTAMP - K_YAWZI_DELAY_S or odom_timestamp + K_YAWZI_DELAY_S

# convert trajectory to GTSAM Pose list
def to_gtsam_poses(T_w_cs: List[np.ndarray]) -> List:
    import gtsam
    poses = []
    for T_w_c in T_w_cs:
        R_w_c = T_w_c[0:3, 0:3]
        t_w_c = T_w_c[0:3, 3]
        pose = gtsam.Pose3(gtsam.Rot3(R_w_c), gtsam.Point3(t_w_c))
        poses.append(pose)
    return poses

def write_tum_file(tum_fp, T_world_cams, timestamps):
    with open(tum_fp, 'w') as f:
        for T_world_cam, timestamp in zip(T_world_cams, timestamps):
            rot = T_world_cam[:3, :3]
            translation = T_world_cam[:3, 3]
            quat = R.from_matrix(rot).as_quat()  # x, y, z, w
            f.write(f"{timestamp:.6f} {translation[0]:.6f} {translation[1]:.6f} {translation[2]:.6f} {quat[0]:.6f} {quat[1]:.6f} {quat[2]:.6f} {quat[3]:.6f}\n")

def get_resized_color(frame_reader, img):
    return cv2.resize(img, (frame_reader.W_out_with_edge, frame_reader.H_out_with_edge))

def main(dataset_name, exp_scene, write_images: bool = False):
    full_resol = False

    if "tektite" in dataset_name:
        site_str = "tektite"
    elif "yawzi" in dataset_name:
        site_str = "yawzi"
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
    code_dir = Path(__file__).parent.parent.parent
    reference_tum = str(code_dir / f"mast3r/datasets/{site_str}/slam/reference_tum_umeyama.tum")

    # Global metrics initialization to save time and VRAM
    ssim_net = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    psnr_net = PeakSignalNoiseRatio(data_range=1.0).to(device)
    lpips_net = LearnedPerceptualImagePatchSimilarity(
        net_type="alex", normalize=True
    ).to(device)

    background = torch.tensor([0, 0.0, 0], dtype=torch.float32, device="cuda")
    all_psnr, all_ssim, all_lpips = [], [], []

    # setup
    exp_folder = f"./output/{dataset_name}/{exp_scene}"
    save_folder = f"{exp_folder}/nvs"
    os.makedirs(os.path.join(save_folder, 'vis'), exist_ok=True)

    with open(os.path.join(exp_folder, 'cfg.yaml'), 'r') as file:
        cfg = yaml.safe_load(file)

    frame_reader = get_dataset(cfg, device='cuda')
    with open(os.path.join(exp_folder, 'traj/est_poses_full.txt'), 'r') as f:
        est_lines = f.readlines()

    traj_T_w_c = []
    timestamps_s = []
    assert len(est_lines) == len(frame_reader.color_paths)
    for idx, (line, color_path) in enumerate(zip(est_lines, frame_reader.color_paths)):
        T_w_c = line_to_T(line)
        traj_T_w_c.append(T_w_c)

        timestamp_s, timestamp_ns = Path(color_path).stem.split('-')
        t_s = float(timestamp_s) + float(timestamp_ns) * 1e-9
        t_s -= K_TEKTITE_DELAY_S if site_str == "tektite" else K_YAWZI_DELAY_S
        timestamps_s.append(t_s)

    rot_wxyz = [R.from_matrix(T_world_cam[:3, :3]).as_quat(scalar_first=True) for T_world_cam in traj_T_w_c]
    pos_xyz = [T_world_cam[:3, 3] for T_world_cam in traj_T_w_c]

    slam_evaluator = EVOEvaluator()
    gtsam_poses = to_gtsam_poses(traj_T_w_c)
    write_tum_file(f"{site_str}_wildgsslam.txt", traj_T_w_c, timestamps_s)
    slam_metrics_from_wildgs = slam_evaluator.evaluate_trajectories(
        reference_tum=reference_tum,
        estimated_poses=gtsam_poses,
        timestamps=timestamps_s,
        metric="ape",
        plot=False,
        save_results=False,
        t_offset=0.0,
        t_max_offset=0.1,
        align=True,
        correct_scale=True,
    )
    print(f"SLAM metrics from saved img poses, reference: {reference_tum}:")
    for k, v in slam_metrics_from_wildgs.items():
        print(f"{k}: {v}")
    rmse = slam_metrics_from_wildgs['rmse']

    batched_wxyzs = np.stack(rot_wxyz)
    batched_positions = np.stack(pos_xyz)

    # Get the model
    gaussians = GaussianModel(0, config=None)
    gaussians.load_ply(os.path.join(exp_folder, 'final_gs.ply'))

    means = gaussians.get_xyz
    quats = gaussians.get_rotation
    scales = gaussians.get_scaling
    opacities = gaussians.get_opacity.squeeze()
    colors = gaussians.get_features
    sh_degree = int(math.sqrt(colors.shape[-2]) - 1)

    # register and open viewer
    @torch.no_grad()
    def viewer_render_fn(camera_state: nerfview.CameraState, img_wh: Tuple[int, int]):
        width, height = img_wh
        c2w = camera_state.c2w
        K = camera_state.get_K(img_wh)
        c2w = torch.from_numpy(c2w).float().to(device)
        K = torch.from_numpy(K).float().to(device)
        viewmat = c2w.inverse()

        rasterization_fn = rasterization

        render_colors, render_alphas, meta = rasterization_fn(
            means,  # [N, 3]
            quats,  # [N, 4]
            scales,  # [N, 3]
            opacities,  # [N]
            colors,  # [N, S, 3]
            viewmat[None],  # [1, 4, 4]
            K[None],  # [1, 3, 3]
            width,
            height,
            sh_degree=sh_degree,
            render_mode="RGB",
            # this is to speedup large-scale rendering by skipping far-away Gaussians.
            radius_clip=0.0,
            backgrounds=torch.ones(1, 3).to(device),  # white background
        )
        render_rgbs = render_colors[0, ..., 0:3].cpu().numpy()
        return render_rgbs

    server = viser.ViserServer(verbose=False)
    _ = nerfview.Viewer(
        server=server,
        render_fn=viewer_render_fn,
        mode="rendering",
    )
    server.scene.add_batched_axes(
        name="trajectory",
        batched_wxyzs=batched_wxyzs,
        batched_positions=batched_positions,
        axes_length=0.5,
        axes_radius=0.05,
        visible=True,
    )

    intrinsics = {
        "fx": cfg['cam']['fx'],
        "fy": cfg['cam']['fy'],
        "ppx": cfg['cam']['cx'],
        "ppy": cfg['cam']['cy'],
        "width": cfg['cam']['W'],
        "height": cfg['cam']['H'],
    }

    pipe = get_render_pipline_params(exp_folder)
    psnr_array, ssim_array, lpips_array = [], [], []

    num_images = len(est_lines)
    for i in tqdm(range(num_images)):
        T_w_c = line_to_T(est_lines[i])
        T_c_w = torch.tensor(np.linalg.inv(T_w_c), device='cuda', dtype=torch.float32)

        viewpoint = get_temp_viewpoint(intrinsics, full_resol=full_resol, exp_cfg=cfg)
        viewpoint.update_RT(T_c_w[:3, :3], T_c_w[:3, 3])
        with torch.no_grad():
            rendering_pkg = render(viewpoint, gaussians, pipe, background)
            image = torch.clamp(rendering_pkg["render"], 0.0, 1.0)

        # Ground Truth Processing
        input_rgb = cv2.imread(frame_reader.color_paths[i])
        input_rgb = get_resized_color(frame_reader, input_rgb)

        # Metrics preparation
        gt_image = torch.from_numpy(input_rgb).float().permute(2, 0, 1).to('cuda') / 255.0

        # Metric calculation
        psnr_score = psnr_net(image.unsqueeze(0), gt_image.unsqueeze(0)).item()
        ssim_score = ssim_net(image.unsqueeze(0), gt_image.unsqueeze(0)).item()
        lpips_score = lpips_net(image.unsqueeze(0), gt_image.unsqueeze(0)).item()

        psnr_array.append(psnr_score)
        ssim_array.append(ssim_score)
        lpips_array.append(lpips_score)

        # Visualization
        rendered_rgb = (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        rendered_rgb_bgr = cv2.cvtColor(rendered_rgb, cv2.COLOR_RGB2BGR)

        if write_images:
            output_img = np.zeros((input_rgb.shape[0] * 2 + 20, input_rgb.shape[1], 3), dtype=np.uint8)
            output_img[:input_rgb.shape[0], :, :] = input_rgb
            output_img[input_rgb.shape[0] + 20:, :, :] = rendered_rgb_bgr
            cv2.imwrite(os.path.join(save_folder, f'vis/nvs_{i:04d}.png'), output_img)

    output = {
        "mean_psnr": float(np.mean(psnr_array)),
        "mean_ssim": float(np.mean(ssim_array)),
        "mean_lpips": float(np.mean(lpips_array)),
        "rmse": rmse,
    }

    print(f'Scene {exp_scene} - Mean PSNR: {output["mean_psnr"]:.4f}, SSIM: {output["mean_ssim"]:.4f}, LPIPS: {output["mean_lpips"]:.4f}')
    json.dump(output, open(os.path.join(save_folder, "final_result.json"), "w"), indent=4)

    return output

if __name__ == "__main__":
    dset_scenes = [
        ("tektite_2x_cc_ss2_reference_images", "tektite_2x_cc_ss1_reference_images"),
        ("yawzi_2x_cc_ss3_reference_images", "yawzi_2x_cc_ss1_reference_images"),
    ]
    res_tektite = main(dset_scenes[0][0], dset_scenes[0][1])
    res_yawzi = main(dset_scenes[1][0], dset_scenes[1][1])

    print(f'Scene tektite - Mean PSNR: {res_tektite["mean_psnr"]:.4f}, SSIM: {res_tektite["mean_ssim"]:.4f}, LPIPS: {res_tektite["mean_lpips"]:.4f}, RMSE: {res_tektite["rmse"]:.4f}')
    print(f'Scene yawzi - Mean PSNR: {res_yawzi["mean_psnr"]:.4f}, SSIM: {res_yawzi["mean_ssim"]:.4f}, LPIPS: {res_yawzi["mean_lpips"]:.4f}, RMSE: {res_yawzi["rmse"]:.4f}')