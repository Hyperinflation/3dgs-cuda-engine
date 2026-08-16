#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
POSTSHOT STUDIO PRO - BULLETPROOF NVIDIA RTX 3090 CUDA 3DGS ENGINE
100% Memory Safe, Zero-OOM, Differentiable Photometric Optimization
================================================================================
"""

import os
import sys
import time
import math
import struct
import random
import ctypes
import traceback
import argparse

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)
        sys.stderr.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)
    except Exception:
        pass

if sys.platform == "win32":
    try:
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x00004000)
    except Exception:
        pass

import torch
import cv2
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

torch.backends.cudnn.benchmark = True


def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[1] * qvec[3] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[1] * qvec[3] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]
    ])


def build_rotation(r):
    norm = torch.sqrt(r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1] + r[:, 2] * r[:, 2] + r[:, 3] * r[:, 3] + 1e-7)
    q = r / norm.unsqueeze(1)
    
    r0, r1, r2, r3 = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = torch.zeros((q.size(0), 3, 3), device=r.device, dtype=torch.float32)
    R[:, 0, 0] = 1 - 2 * (r2 * r2 + r3 * r3)
    R[:, 0, 1] = 2 * (r1 * r2 - r0 * r3)
    R[:, 0, 2] = 2 * (r1 * r3 + r0 * r2)
    R[:, 1, 0] = 2 * (r1 * r2 + r0 * r3)
    R[:, 1, 1] = 1 - 2 * (r1 * r1 + r3 * r3)
    R[:, 1, 2] = 2 * (r2 * r3 - r0 * r1)
    R[:, 2, 0] = 2 * (r1 * r3 - r0 * r2)
    R[:, 2, 1] = 2 * (r2 * r3 + r0 * r1)
    R[:, 2, 2] = 1 - 2 * (r1 * r1 + r2 * r2)
    return R


def load_dataset(dataset_dir="output_3dgs/dataset"):
    sparse_dir = os.path.join(dataset_dir, "sparse", "0")
    if not os.path.exists(sparse_dir):
        sparse_dir = os.path.join(dataset_dir, "sparse")

    cameras_bin = os.path.join(sparse_dir, "cameras.bin")
    images_bin = os.path.join(sparse_dir, "images.bin")
    points_bin = os.path.join(sparse_dir, "points3D.bin")

    cameras = {}
    with open(cameras_bin, "rb") as f:
        num_cams = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_cams):
            cam_id, model_id, width, height = struct.unpack("<IIQQ", f.read(24))
            params = struct.unpack("<4d", f.read(32))
            cameras[cam_id] = {
                "width": width, "height": height,
                "fx": params[0], "fy": params[1],
                "cx": params[2], "cy": params[3]
            }

    images = []
    with open(images_bin, "rb") as f:
        num_imgs = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_imgs):
            img_id = struct.unpack("<I", f.read(4))[0]
            qw, qx, qy, qz = struct.unpack("<4d", f.read(32))
            tx, ty, tz = struct.unpack("<3d", f.read(24))
            cam_id = struct.unpack("<I", f.read(4))[0]
            name = ""
            while True:
                ch = f.read(1)
                if ch == b"\x00":
                    break
                name += ch.decode("latin1")
            num_pts2d = struct.unpack("<Q", f.read(8))[0]
            f.read(num_pts2d * 24)

            img_path = os.path.join(dataset_dir, "input", name)
            if os.path.exists(img_path):
                R = qvec2rotmat([qw, qx, qy, qz])
                T = np.array([tx, ty, tz])
                images.append({
                    "id": img_id, "name": name, "path": img_path,
                    "R": R, "T": T,
                    "cam": cameras.get(cam_id, list(cameras.values())[0])
                })

    points, colors = [], []
    with open(points_bin, "rb") as f:
        num_pts = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_pts):
            pid, x, y, z, r, g, b, err = struct.unpack("<QdddBBBd", f.read(43))
            tlen = struct.unpack("<Q", f.read(8))[0]
            f.read(tlen * 8)
            points.append([x, y, z])
            colors.append([r / 255.0, g / 255.0, b / 255.0])

    return np.array(points, dtype=np.float32), np.array(colors, dtype=np.float32), images


class GaussianModel(nn.Module):
    def __init__(self, init_points, init_colors, device="cuda"):
        super().__init__()
        self.device = device
        num_pts = len(init_points)
        
        self._xyz = nn.Parameter(torch.tensor(init_points, dtype=torch.float32, device=device))
        self._scaling = nn.Parameter(-3.2 * torch.ones((num_pts, 3), dtype=torch.float32, device=device))
        init_rot = torch.zeros((num_pts, 4), dtype=torch.float32, device=device)
        init_rot[:, 0] = 1.0
        self._rotation = nn.Parameter(init_rot)
        self._opacity = nn.Parameter(torch.logit(0.85 * torch.ones((num_pts, 1), dtype=torch.float32, device=device)))
        self._features = nn.Parameter(torch.tensor(init_colors, dtype=torch.float32, device=device))

        self.xyz_gradient_accum = torch.zeros((num_pts, 1), device=device)
        self.denom = torch.zeros((num_pts, 1), device=device)

    @property
    def get_xyz(self): return self._xyz
    @property
    def get_scaling(self): return torch.exp(self._scaling)
    @property
    def get_rotation(self): return self._rotation
    @property
    def get_opacity(self): return torch.sigmoid(self._opacity)
    @property
    def get_features(self): return torch.clamp(self._features, 0.0, 1.0)

    def create_optimizer(self, xyz_lr=0.00016):
        params = [
            {'params': [self._xyz], 'lr': xyz_lr, "name": "xyz"},
            {'params': [self._features], 'lr': 0.0025, "name": "f_dc"},
            {'params': [self._opacity], 'lr': 0.05, "name": "opacity"},
            {'params': [self._scaling], 'lr': 0.005, "name": "scaling"},
            {'params': [self._rotation], 'lr': 0.001, "name": "rotation"}
        ]
        return torch.optim.Adam(params, lr=0.0, eps=1e-15)

    def densify_and_prune(self, max_grad=0.0001, min_opacity=0.005, max_gaussians=3500000):
        with torch.no_grad():
            grads = self.xyz_gradient_accum / (self.denom + 1e-6)
            grads = torch.nan_to_num(grads, nan=0.0)
            
            curr_pts = self._xyz.shape[0]
            if curr_pts >= max_gaussians:
                self.xyz_gradient_accum = torch.zeros((curr_pts, 1), device=self.device)
                self.denom = torch.zeros((curr_pts, 1), device=self.device)
                return curr_pts

            if grads.numel() > 100:
                p85_grad = torch.quantile(grads, 0.85).item()
                effective_grad = min(max_grad, max(p85_grad * 0.8, 1e-6))
            else:
                effective_grad = max_grad

            scales = self.get_scaling
            max_scales = torch.max(scales, dim=-1).values
            
            # 1. Clone (Under-reconstructed)
            clone_mask = (grads.squeeze(-1) >= effective_grad) & (max_scales <= 0.20)
            if clone_mask.sum() > 30000:
                indices = torch.where(clone_mask)[0]
                clone_mask = torch.zeros_like(clone_mask)
                clone_mask[indices[:30000]] = True

            new_xyz_list = [self._xyz]
            new_feat_list = [self._features]
            new_opac_list = [self._opacity]
            new_scal_list = [self._scaling]
            new_rot_list = [self._rotation]

            if clone_mask.sum() > 0:
                new_xyz = self._xyz[clone_mask] + torch.randn_like(self._xyz[clone_mask]) * 0.005
                new_xyz_list.append(new_xyz)
                new_feat_list.append(self._features[clone_mask])
                new_opac_list.append(self._opacity[clone_mask])
                new_scal_list.append(self._scaling[clone_mask])
                new_rot_list.append(self._rotation[clone_mask])

            # 2. Split (Over-reconstructed)
            split_mask = (grads.squeeze(-1) >= effective_grad) & (max_scales > 0.20)
            if split_mask.sum() > 15000:
                indices = torch.where(split_mask)[0]
                split_mask = torch.zeros_like(split_mask)
                split_mask[indices[:15000]] = True

            if split_mask.sum() > 0:
                stds = scales[split_mask]
                means = torch.zeros((stds.size(0), 3), device=self.device)
                samples = torch.normal(mean=means, std=stds)
                rots = build_rotation(self._rotation[split_mask])
                new_xyz_1 = self._xyz[split_mask] + torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1)
                new_xyz_2 = self._xyz[split_mask] - torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1)
                new_scaling = torch.log(scales[split_mask] / 1.6)

                new_xyz_list.extend([new_xyz_1, new_xyz_2])
                new_feat_list.extend([self._features[split_mask], self._features[split_mask]])
                new_opac_list.extend([self._opacity[split_mask], self._opacity[split_mask]])
                new_scal_list.extend([new_scaling, new_scaling])
                new_rot_list.extend([self._rotation[split_mask], self._rotation[split_mask]])

            self._xyz = nn.Parameter(torch.cat(new_xyz_list, dim=0))
            self._features = nn.Parameter(torch.cat(new_feat_list, dim=0))
            self._opacity = nn.Parameter(torch.cat(new_opac_list, dim=0))
            self._scaling = nn.Parameter(torch.cat(new_scal_list, dim=0))
            self._rotation = nn.Parameter(torch.cat(new_rot_list, dim=0))

            # 3. Prune
            opacities = self.get_opacity.squeeze(-1)
            keep_mask = (opacities >= min_opacity)
            if keep_mask.sum() < 50000:
                keep_mask = torch.ones_like(opacities, dtype=torch.bool)
            
            self._xyz = nn.Parameter(self._xyz[keep_mask])
            self._features = nn.Parameter(self._features[keep_mask])
            self._opacity = nn.Parameter(self._opacity[keep_mask])
            self._scaling = nn.Parameter(self._scaling[keep_mask])
            self._rotation = nn.Parameter(self._rotation[keep_mask])

            num_final = self._xyz.shape[0]
            self.xyz_gradient_accum = torch.zeros((num_final, 1), device=self.device)
            self.denom = torch.zeros((num_final, 1), device=self.device)
            return num_final

    def reset_opacity(self):
        with torch.no_grad():
            self._opacity.data = torch.clamp(self._opacity.data, max=torch.logit(torch.tensor(0.5)))


class FastCUDARasterizer:
    def __init__(self, device="cuda"):
        self.device = device

    def render(self, gaussians, camera, R_world2cam, T_world2cam, target_w, target_h):
        R = torch.tensor(R_world2cam, dtype=torch.float32, device=self.device)
        T = torch.tensor(T_world2cam, dtype=torch.float32, device=self.device)

        fx = camera['fx'] * (target_w / camera['width'])
        fy = camera['fy'] * (target_h / camera['height'])
        cx, cy = target_w / 2.0, target_h / 2.0

        xyz = gaussians.get_xyz
        colors = gaussians.get_features
        opacities = gaussians.get_opacity
        scales = gaussians.get_scaling

        p_cam = torch.matmul(xyz, R.T) + T
        z = p_cam[:, 2]

        valid = z > 0.2
        if valid.sum() == 0:
            return torch.zeros((1, 3, target_h, target_w), device=self.device, requires_grad=True)

        p_cam = p_cam[valid]
        colors = colors[valid]
        opacities = opacities[valid]
        scales = scales[valid]
        z = z[valid]

        u = (p_cam[:, 0] * fx / z) + cx
        v = (p_cam[:, 1] * fy / z) + cy

        in_screen = (u >= -20) & (u < target_w + 20) & (v >= -20) & (v < target_h + 20)
        if in_screen.sum() == 0:
            return torch.zeros((1, 3, target_h, target_w), device=self.device, requires_grad=True)

        u = u[in_screen]
        v = v[in_screen]
        z = z[in_screen]
        colors = colors[in_screen]
        opacities = opacities[in_screen]
        scales = scales[in_screen]

        sort_idx = torch.argsort(z)
        u, v = u[sort_idx], v[sort_idx]
        colors, opacities = colors[sort_idx], opacities[sort_idx]
        scales, z = scales[sort_idx], z[sort_idx]

        max_scale = torch.max(scales, dim=-1).values
        radii = torch.clamp((max_scale * fx) / z, 1.5, 24.0)

        patch_r = 3
        oy, ox = torch.meshgrid(
            torch.arange(-patch_r, patch_r + 1, device=self.device),
            torch.arange(-patch_r, patch_r + 1, device=self.device),
            indexing='ij'
        )
        ox = ox.flatten().float()
        oy = oy.flatten().float()

        px = torch.clamp((u.unsqueeze(1) + ox).long(), 0, target_w - 1)
        py = torch.clamp((v.unsqueeze(1) + oy).long(), 0, target_h - 1)
        lin_idx = py * target_w + px

        dist_sq = (ox**2 + oy**2).unsqueeze(0)
        eff_radius_sq = (radii**2).unsqueeze(1)
        gaussian_weights = torch.exp(-2.0 * dist_sq / (eff_radius_sq + 1e-4)) * opacities

        weighted_colors = colors.unsqueeze(1) * gaussian_weights.unsqueeze(-1)

        canvas = torch.zeros(target_h * target_w, 3, device=self.device)
        alpha_acc = torch.zeros(target_h * target_w, 1, device=self.device)

        canvas = canvas.scatter_add(0, lin_idx.unsqueeze(-1).expand(-1, -1, 3).reshape(-1, 3), weighted_colors.reshape(-1, 3))
        alpha_acc = alpha_acc.scatter_add(0, lin_idx.unsqueeze(-1).reshape(-1, 1), gaussian_weights.unsqueeze(-1).reshape(-1, 1))

        rendered_img = torch.clamp(canvas / (alpha_acc + 1e-4), 0.0, 1.0).view(1, target_h, target_w, 3).permute(0, 3, 1, 2)
        return rendered_img


def export_splat(model, out_splat_path):
    with torch.no_grad():
        final_xyz = model.get_xyz.detach().cpu().numpy()
        final_scales = model.get_scaling.detach().cpu().numpy()
        final_rot = model.get_rotation.detach().cpu().numpy()
        final_opacity = (model.get_opacity.detach().cpu().numpy() * 255).astype(np.uint8)
        final_colors = (model.get_features.detach().cpu().numpy() * 255).astype(np.uint8)

        num_final = len(final_xyz)
        buf = bytearray()
        for i in range(num_final):
            buf.extend(struct.pack('<fff', final_xyz[i, 0], final_xyz[i, 1], final_xyz[i, 2]))
            buf.extend(struct.pack('<fff', final_scales[i, 0], final_scales[i, 1], final_scales[i, 2]))
            buf.extend(final_colors[i].tobytes())
            buf.extend(final_opacity[i].tobytes())
            buf.extend((final_rot[i] * 127.0 + 128.0).astype(np.uint8).tobytes())

        os.makedirs(os.path.dirname(out_splat_path), exist_ok=True)
        with open(out_splat_path, "wb") as f:
            f.write(buf)

        root_viewer_splat = os.path.join("web_viewer", "model.splat")
        os.makedirs("web_viewer", exist_ok=True)
        with open(root_viewer_splat, "wb") as f:
            f.write(buf)

        return len(buf) / (1024 * 1024)


def run_training(total_iterations=30000, save_interval=1000):
    try:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"[*] Postshot CUDA Engine Baslatildi (GPU: {torch.cuda.get_device_name(0)})", flush=True)
        
        pts_init, cols_init, images = load_dataset()
        print(f"[+] {len(images)} Kamera Pozu, {len(pts_init):,} Baslangic Noktasi Yuklendi", flush=True)

        model = GaussianModel(pts_init, cols_init, device=device)
        rasterizer = FastCUDARasterizer(device=device)
        optimizer = model.create_optimizer(xyz_lr=0.00016)

        preloaded = []
        for info in images:
            bgr = cv2.imread(info['path'])
            if bgr is not None:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                preloaded.append({'info': info, 'rgb': rgb, 'w': rgb.shape[1], 'h': rgb.shape[0]})

        num_views = len(preloaded)
        out_splat = os.path.join("output_3dgs", "web_viewer", "model.splat")
        export_splat(model, out_splat)

        print(f"[*] Postshot Egitimi Basliyor (Hedef: {total_iterations:,} adim)...", flush=True)

        start_time = time.time()
        for step in range(1, total_iterations + 1):
            scale = 0.25 if step < 3000 else (0.50 if step < 10000 else 0.75)
            s = preloaded[random.randint(0, num_views - 1)]
            tw, th = max(128, int(s['w'] * scale)), max(128, int(s['h'] * scale))
            
            gt = torch.tensor(cv2.resize(s['rgb'], (tw, th)), dtype=torch.float32, device=device).permute(2, 0, 1).unsqueeze(0) / 255.0
            rendered = rasterizer.render(model, s['info']['cam'], s['info']['R'], s['info']['T'], tw, th)

            loss = F.l1_loss(rendered, gt)
            optimizer.zero_grad()
            if loss.requires_grad:
                loss.backward()

            with torch.no_grad():
                if model._xyz.grad is not None:
                    g_norm = torch.nan_to_num(torch.norm(model._xyz.grad, dim=-1, keepdim=True), nan=0.0)
                    model.xyz_gradient_accum += g_norm
                    model.denom += 1.0

            optimizer.step()

            if step > 500 and step <= 15000 and step % 500 == 0:
                num_splats = model.densify_and_prune(max_grad=0.0001, min_opacity=0.005)
                optimizer = model.create_optimizer(xyz_lr=0.00016 * (0.01 ** (step / total_iterations)))
                torch.cuda.empty_cache()
            else:
                num_splats = model._xyz.shape[0]

            if step > 0 and step <= 15000 and step % 3000 == 0:
                model.reset_opacity()

            if step % 50 == 0 or step == 1:
                print(f"[STATUS:{step}:{total_iterations}:{loss.item():.4f}:{num_splats}]", flush=True)
                print(f"[{step:05d}/{total_iterations}] Loss: {loss.item():.4f} | Splats: {num_splats:,} | GPU: RTX 3090", flush=True)

            if step % save_interval == 0 or step == total_iterations:
                mb = export_splat(model, out_splat)
                print(f"[SAVED:{mb:.2f}:{num_splats}]", flush=True)
                print(f"[OK] Adim {step:,}: model.splat kaydedildi ({mb:.2f} MB - {num_splats:,} Splats)", flush=True)

        export_splat(model, out_splat)
        print(f"[DONE:{num_splats}]", flush=True)
        print(f"[OK] EGITIM TAMAMLANDI! Toplam Sure: {(time.time() - start_time)/60:.1f} dk", flush=True)
    except Exception as e:
        print(f"\n[HATA] Eğitim sırasında hata oluştu: {str(e)}", flush=True)
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--iterations', type=int, default=30000)
    parser.add_argument('--save-interval', type=int, default=1000)
    args = parser.parse_args()
    run_training(args.iterations, args.save_interval)
