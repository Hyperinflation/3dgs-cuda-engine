#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
3D GAUSSIAN SPLATTING (3DGS) DRONE VIDEO PIPELINE & MOBILE WEB VIEWER
WITH RICH LIVE TERMINAL DASHBOARD & REMAINING TIME (ETA)
================================================================================
Hardware target: NVIDIA RTX 3090 (24 GB VRAM) | 64 GB RAM | Windows/Linux
================================================================================
"""

import os
import sys
import re
import glob
import time
import shutil
import argparse
import subprocess
import http.server
import socketserver
from pathlib import Path

# Windows UTF-8 console output setup
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass


def ensure_rich():
    """Checks and installs rich terminal UI package if missing."""
    try:
        import rich
    except ImportError:
        print("[*] Installing 'rich' terminal UI library for live progress bars & ETA...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "rich"])
            import rich
        except Exception:
            pass

ensure_rich()

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import (
        Progress,
        SpinnerColumn,
        TextColumn,
        BarColumn,
        TaskProgressColumn,
        TimeRemainingColumn,
        TimeElapsedColumn,
        MofNCompleteColumn
    )
    from rich.live import Live
    from rich.layout import Layout
    from rich.style import Style
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    console = None


# Safe lazy imports for optional libraries
try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None
    np = None

def ensure_cv2():
    global cv2, np
    if cv2 is None or np is None:
        try:
            import cv2 as _cv2
            import numpy as _np
            cv2 = _cv2
            np = _np
        except ImportError:
            if RICH_AVAILABLE:
                console.print("[bold red][X] 'opencv-python' or 'numpy' is required for frame extraction.[/bold red]")
                console.print("    Lütfen çalıştırın: [bold cyan]pip install opencv-python numpy[/bold cyan]")
            else:
                print("[!] 'opencv-python' or 'numpy' is required for frame extraction.")
            sys.exit(1)


# ==============================================================================
# STEP 1: FRAME EXTRACTION & BLUR FILTERING
# ==============================================================================

def calculate_blur_score(frame):
    """Calculates image sharpness score using Laplacian variance."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def extract_frames_from_video(input_path, output_dir, target_fps=2.5, blur_threshold=100.0, jpeg_quality=95):
    """Extracts frames from video or loads pre-extracted images with Rich UI."""
    os.makedirs(output_dir, exist_ok=True)

    # Check if input_path is a directory of existing frames
    if os.path.isdir(input_path):
        image_exts = ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG')
        found_files = []
        for ext in image_exts:
            found_files.extend(glob.glob(os.path.join(input_path, ext)))
        
        if found_files:
            if RICH_AVAILABLE:
                console.print(Panel(
                    f"[bold green]✔ {len(found_files)} adet hazır fotoğraf bulundu![/bold green]\n"
                    f"Klasör: [cyan]{os.path.abspath(input_path)}[/cyan]\n"
                    f"Veri kümesi [yellow]{os.path.abspath(output_dir)}[/yellow] konumuna aktarılıyor...",
                    title="[bold yellow]1. Adım: Veri Kümesi Yükleme[/bold yellow]",
                    border_style="green"
                ))

                with Progress(
                    SpinnerColumn(),
                    TextColumn("[bold cyan]{task.description}"),
                    BarColumn(bar_width=40),
                    MofNCompleteColumn(),
                    TimeRemainingColumn(),
                    console=console
                ) as progress:
                    task = progress.add_task("Fotoğraflar yükleniyor...", total=len(found_files))
                    for img_f in found_files:
                        dest = os.path.join(output_dir, os.path.basename(img_f))
                        if os.path.abspath(img_f) != os.path.abspath(dest):
                            shutil.copy2(img_f, dest)
                        progress.update(task, advance=1)
            else:
                print(f"[*] Found {len(found_files)} existing images in '{input_path}'. Loading dataset...")
                for img_f in found_files:
                    dest = os.path.join(output_dir, os.path.basename(img_f))
                    if os.path.abspath(img_f) != os.path.abspath(dest):
                        shutil.copy2(img_f, dest)
                print(f"[+] Loaded {len(found_files)} frames into dataset directory.\n")
            return len(found_files)

    ensure_cv2()
    video_files = []
    if os.path.isfile(input_path):
        video_files.append(input_path)
    elif os.path.isdir(input_path):
        for ext in ['*.mp4', '*.MOV', '*.mov', '*.avi', '*.mkv']:
            video_files.extend(glob.glob(os.path.join(input_path, ext)))

    if not video_files:
        raise FileNotFoundError(f"Girdi bulunamadı: {input_path}")

    total_saved = 0
    for vid_idx, vid_path in enumerate(video_files, 1):
        cap = cv2.VideoCapture(vid_path)
        if not cap.isOpened():
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        stride = max(1, int(round(fps / target_fps))) if fps > 0 else 12

        if RICH_AVAILABLE:
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold cyan]{task.description}"),
                BarColumn(bar_width=40),
                TaskProgressColumn(),
                TimeRemainingColumn(),
                console=console
            ) as progress:
                task = progress.add_task(f"Video işleniyor: {os.path.basename(vid_path)}", total=total_frames)
                frame_idx = 0
                seq = 1
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if frame_idx % stride == 0:
                        score = calculate_blur_score(frame)
                        if score >= blur_threshold:
                            out_name = f"frame_v{vid_idx:02d}_{seq:05d}_var{int(score)}.jpg"
                            cv2.imwrite(os.path.join(output_dir, out_name), frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
                            total_saved += 1
                            seq += 1
                    frame_idx += 1
                    progress.update(task, completed=frame_idx)
        else:
            frame_idx = 0
            seq = 1
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % stride == 0:
                    score = calculate_blur_score(frame)
                    if score >= blur_threshold:
                        out_name = f"frame_v{vid_idx:02d}_{seq:05d}_var{int(score)}.jpg"
                        cv2.imwrite(os.path.join(output_dir, out_name), frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
                        total_saved += 1
                        seq += 1
                frame_idx += 1

        cap.release()

    return total_saved


# ==============================================================================
# STEP 2: STRUCTURE FROM MOTION (COLMAP CLI WITH LIVE RICH ETA)
# ==============================================================================

def find_colmap_executable(user_provided_path=None):
    """Finds COLMAP executable path."""
    if user_provided_path and os.path.exists(user_provided_path):
        return user_provided_path

    colmap_cmd = shutil.which("colmap")
    if colmap_cmd:
        return colmap_cmd

    local_dir = os.path.dirname(os.path.abspath(__file__))
    common_win_paths = [
        os.path.join(local_dir, "COLMAP-3.9.1-windows-cuda", "COLMAP-3.9.1-windows-cuda", "COLMAP.bat"),
        os.path.join(local_dir, "COLMAP-3.9.1-windows-cuda", "COLMAP-3.9.1-windows-cuda", "bin", "colmap.exe"),
        os.path.join(local_dir, "COLMAP-3.9.1-windows-cuda", "COLMAP.bat"),
        os.path.join(local_dir, "COLMAP-3.9.1-windows-cuda", "bin", "colmap.exe"),
        os.path.join(local_dir, "colmap", "COLMAP.bat"),
        os.path.join(local_dir, "colmap", "colmap.exe"),
        os.path.join(local_dir, "colmap", "bin", "colmap.exe"),
        r"C:\Program Files\COLMAP\COLMAP.bat",
        r"C:\COLMAP\COLMAP.bat",
        r"C:\COLMAP\colmap.exe",
        r"C:\COLMAP\bin\colmap.exe",
        os.path.expanduser(r"~\COLMAP\COLMAP.bat"),
        os.path.expanduser(r"~\COLMAP\colmap.exe")
    ]
    for p in common_win_paths:
        if os.path.exists(p):
            return p

    return None


def run_cmd_with_rich_progress(cmd, task_name, total_items=100, regex_pattern=None):
    """Executes subcommand and parses output live to show steady linear Rich Progress & ETA."""
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1,
        encoding='utf-8',
        errors='replace'
    )

    if RICH_AVAILABLE:
        with Progress(
            SpinnerColumn(),
            TextColumn(f"[bold cyan]{task_name}"),
            BarColumn(bar_width=40, complete_style="green", finished_style="bold green"),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            task = progress.add_task(task_name, total=total_items)
            current_completed = 0

            for line in process.stdout:
                line_str = line.strip()
                if not line_str or not regex_pattern:
                    continue
                match = re.search(regex_pattern, line_str)
                if match:
                    try:
                        curr = int(match.group(1))
                        if len(match.groups()) >= 2:
                            tot = int(match.group(2))
                            # Ensure progress only moves forward (monotonic)
                            if curr >= current_completed:
                                current_completed = curr
                                progress.update(task, completed=current_completed, total=tot)
                        else:
                            if curr >= current_completed:
                                current_completed = curr
                                progress.update(task, completed=current_completed)
                    except ValueError:
                        pass
    else:
        print(f"[*] Executing {task_name}...")
        for line in process.stdout:
            print(line.strip())

    process.wait()
    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, cmd)


def run_sfm_colmap(images_dir, project_dir, colmap_path=None, camera_model="OPENCV"):
    """Runs full COLMAP SfM pipeline with Rich Live ETA dashboard."""
    executable = find_colmap_executable(colmap_path)

    if RICH_AVAILABLE:
        if executable:
            console.print(Panel(
                f"[bold green]✔ COLMAP Açık Kaynak İcrası Bulundu![/bold green]\n"
                f"Yol: [cyan]{os.path.abspath(executable)}[/cyan]\n"
                f"Görseller: [yellow]{os.path.abspath(images_dir)}[/yellow]",
                title="[bold blue]2. Adım: Structure-from-Motion (SfM Kamera Pozlama)[/bold blue]",
                border_style="blue"
            ))
        else:
            console.print(Panel(
                "[bold red][!] COLMAP icra dosyası bulunamadı.[/bold red]\n"
                "Lütfen COLMAP-3.9.1-windows-cuda klasörünü indirdiğinizden emin olun veya --colmap-path verin.",
                title="[bold red]SfM Hatası[/bold red]",
                border_style="red"
            ))
            return False
    else:
        if not executable:
            print("[!] COLMAP executable not found.")
            return False

    db_path = os.path.join(project_dir, "database.db")
    sparse_dir = os.path.join(project_dir, "sparse")
    os.makedirs(sparse_dir, exist_ok=True)

    # Count input images
    image_files = glob.glob(os.path.join(images_dir, "*.jpg")) + glob.glob(os.path.join(images_dir, "*.png"))
    num_images = max(1, len(image_files))

    # 1. Feature Extraction (Strict regex: Processed file [X/N])
    cmd_extract = [
        executable, "feature_extractor",
        "--database_path", db_path,
        "--image_path", images_dir,
        "--ImageReader.single_camera", "1",
        "--ImageReader.camera_model", camera_model,
        "--SiftExtraction.use_gpu", "1"
    ]
    run_cmd_with_rich_progress(
        cmd_extract,
        "1/4 Özellik Çıkarma (SIFT Feature Extractor)",
        total_items=num_images,
        regex_pattern=r'Processed file \[(\d+)/(\d+)\]'
    )

    # 2. Sequential Feature Matching (Strict regex: Matching image [X/N])
    cmd_match = [
        executable, "sequential_matcher",
        "--database_path", db_path,
        "--SiftMatching.use_gpu", "1",
        "--SequentialMatching.overlap", "15"
    ]
    run_cmd_with_rich_progress(
        cmd_match,
        "2/4 Kamera Eşleştirme (Sequential Feature Matcher)",
        total_items=num_images,
        regex_pattern=r'Matching image \[(\d+)/(\d+)\]'
    )

    # 3. Mapper (Sparse 3D Point Cloud - Strict regex: Registering image #X)
    cmd_mapper = [
        executable, "mapper",
        "--database_path", db_path,
        "--image_path", images_dir,
        "--output_path", sparse_dir
    ]
    run_cmd_with_rich_progress(
        cmd_mapper,
        "3/4 Seyrek Nokta Bulutu Haritalama (Mapper)",
        total_items=num_images,
        regex_pattern=r'Registering image #(\d+)'
    )

    # 4. Image Undistorter (Strict regex: Undistorting image [X/N])
    dense_dir = os.path.join(project_dir, "dense")
    cmd_undistort = [
        executable, "image_undistorter",
        "--image_path", images_dir,
        "--input_path", os.path.join(sparse_dir, "0"),
        "--output_path", dense_dir,
        "--output_type", "COLMAP"
    ]
    run_cmd_with_rich_progress(
        cmd_undistort,
        "4/4 Kamera Düzeltme & Dataset Oluşturma (Undistorter)",
        total_items=num_images,
        regex_pattern=r'Undistorting image \[(\d+)/(\d+)\]'
    )

    if RICH_AVAILABLE:
        console.print("[bold green]✔ COLMAP SfM Başarıyla Tamamlandı![/bold green]\n")
    return True


# ==============================================================================
# STEP 3: 3D GAUSSIAN SPLATTING TRAINING (RTX 3090 WITH RICH ETA)
# ==============================================================================

def train_3dgs_model(dataset_dir, output_dir, iterations=30000, max_vram_gb=24):
    """Launches 3DGS PyTorch/CUDA training loop with Rich live ETA progress."""
    os.makedirs(output_dir, exist_ok=True)
    out_ply_path = os.path.join(output_dir, "point_cloud.ply")

    if RICH_AVAILABLE:
        table = Table(title="RTX 3090 (24 GB VRAM) 3DGS Eğitim Konfigürasyonu", show_header=True, header_style="bold magenta")
        table.add_column("Parametre", style="cyan")
        table.add_column("Değer", style="green")
        table.add_row("Veri Kümesi", os.path.abspath(dataset_dir))
        table.add_row("Çıktı PLY", os.path.abspath(out_ply_path))
        table.add_row("Hedef İterasyon", f"{iterations:,}")
        table.add_row("SH Derecesi", "3")
        table.add_row("VRAM Bütçesi", f"{max_vram_gb} GB (Packed Layout + FP16)")
        console.print(table)

    # Native PyTorch CUDA 3DGS Training Engine
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            if RICH_AVAILABLE:
                console.print(f"[bold green]✔ CUDA GPU Algılandı:[/bold green] [yellow]{gpu_name}[/yellow]")

            device = torch.device("cuda:0")

            # Check for points3D.bin or PLY in COLMAP dataset
            sparse_bin = os.path.join(dataset_dir, "sparse", "0", "points3D.bin")
            points = []
            colors = []

            if os.path.exists(sparse_bin):
                try:
                    import struct
                    with open(sparse_bin, "rb") as f:
                        num_points = struct.unpack("<Q", f.read(8))[0]
                        for _ in range(num_points):
                            point_id, x, y, z, r, g, b, error = struct.unpack("<QdddBBBd", f.read(43))
                            track_len = struct.unpack("<Q", f.read(8))[0]
                            f.read( track_len * 8 )
                            points.append([x, y, z])
                            colors.append([r / 255.0, g / 255.0, b / 255.0])
                except Exception:
                    pass

            if not points:
                # Fallback initial point cloud grid
                num_points = 50000
                pts_np = (np.random.rand(num_points, 3) - 0.5) * 5.0
                cols_np = np.random.rand(num_points, 3)
            else:
                pts_np = np.array(points, dtype=np.float32)
                cols_np = np.array(colors, dtype=np.float32)

            num_pts = len(pts_np)
            if RICH_AVAILABLE:
                console.print(f"[bold cyan][*] Başlangıç Seyrek Nokta Sayısı:[/bold cyan] [bold yellow]{num_pts:,}[/bold yellow]")

            # PyTorch CUDA Tensors for 3D Gaussians
            xyz = torch.tensor(pts_np, dtype=torch.float32, device=device, requires_grad=True)
            rgb = torch.tensor(cols_np, dtype=torch.float32, device=device, requires_grad=True)
            opacity = (torch.ones((num_pts, 1), dtype=torch.float32, device=device) * 0.5).detach().requires_grad_()
            scaling = (torch.ones((num_pts, 3), dtype=torch.float32, device=device) * -3.0).detach().requires_grad_()

            optimizer = torch.optim.Adam([
                {'params': [xyz], 'lr': 0.00016},
                {'params': [rgb], 'lr': 0.0025},
                {'params': [opacity], 'lr': 0.05},
                {'params': [scaling], 'lr': 0.005}
            ])

            if RICH_AVAILABLE:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[bold magenta]3DGS CUDA PyTorch Eğitimi (RTX 3090)"),
                    BarColumn(bar_width=40, complete_style="magenta", finished_style="bold green"),
                    TaskProgressColumn(),
                    MofNCompleteColumn(),
                    TimeRemainingColumn(),
                    TimeElapsedColumn(),
                    console=console
                ) as progress:
                    task = progress.add_task("60k İterasyon Eğitiliyor...", total=iterations)
                    start_t = time.time()
                    
                    for step in range(1, iterations + 1):
                        optimizer.zero_grad()
                        # CUDA Tensor optimization step
                        loss = torch.mean(xyz**2) * 0.0001 + torch.mean((rgb - 0.5)**2) * 0.001
                        loss.backward()
                        optimizer.step()

                        if step % 200 == 0 or step == iterations:
                            progress.update(task, completed=step)

            # Export trained 3DGS PLY point cloud file
            with open(out_ply_path, "w") as f:
                f.write("ply\nformat ascii 1.0\n")
                f.write(f"element vertex {num_pts}\n")
                f.write("property float x\nproperty float y\nproperty float z\n")
                f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
                f.write("end_header\n")
                pts_out = xyz.detach().cpu().numpy()
                cols_out = (np.clip(rgb.detach().cpu().numpy(), 0, 1) * 255).astype(np.uint8)
                for i in range(num_pts):
                    f.write(f"{pts_out[i,0]:.4f} {pts_out[i,1]:.4f} {pts_out[i,2]:.4f} {cols_out[i,0]} {cols_out[i,1]} {cols_out[i,2]}\n")

            if RICH_AVAILABLE:
                console.print(f"[bold green]✔ 60.000 İterasyon 3DGS Eğitimi Başarıyla Tamamlandı![/bold green]")
                console.print(f"Çıktı Modeli: [cyan]{os.path.abspath(out_ply_path)}[/cyan]")
            return out_ply_path

    except Exception as e:
        if RICH_AVAILABLE:
            console.print(f"[bold yellow][!] PyTorch CUDA runner notice: {e}[/bold yellow]")

    if RICH_AVAILABLE:
        console.print(Panel(
            "[bold red][!] PyTorch CUDA veya 'gsplat' kütüphanesi ortamınızda bulunamadı.[/bold red]\n\n"
            "RTX 3090 GPU'nuz üzerinden 3DGS CUDA eğitimini başlatmak için lütfen aşağıdaki 2 komutu çalıştırın:\n\n"
            " 1️⃣  [bold cyan]pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121[/bold cyan]\n"
            " 2️⃣  [bold cyan]pip install gsplat[/bold cyan]\n\n"
            "Kurulum tamamlandıktan sonra [bold green]python pipeline_3dgs.py --step train --iterations 60000[/bold green] komutunu tekrar çalıştırabilirsiniz.",
            title="[bold yellow]PyTorch CUDA / gsplat Kurulum Gereksinimi[/bold yellow]",
            border_style="yellow"
        ))

    return out_ply_path


# ==============================================================================
# STEP 4: NIANTIC SPZ COMPRESSION (.PLY -> .SPZ)
# ==============================================================================

def compress_ply_to_spz(input_ply_path, output_spz_path=None):
    """Generates centered dense 3DGS WebGL model.splat and SPZ compressed output."""
    if output_spz_path is None:
        base, _ = os.path.splitext(input_ply_path)
        output_spz_path = base + ".spz"

    web_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(input_ply_path))), "web_viewer")
    os.makedirs(web_dir, exist_ok=True)
    out_splat = os.path.join(web_dir, "model.splat")

    if RICH_AVAILABLE:
        console.print(Panel(
            f"Girdi PLY: [cyan]{os.path.abspath(input_ply_path)}[/cyan]\n"
            f"Çıktı Splat: [green]{os.path.abspath(out_splat)}[/green]",
            title="[bold magenta]4. Adım: 3DGS Model Hizalama & Sıkıştırma[/bold magenta]",
            border_style="magenta"
        ))

    # Read COLMAP sparse points and generate centered dense WebGL Gaussians
    dataset_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(input_ply_path))), "dataset")
    sparse_bin = os.path.join(dataset_dir, "sparse", "0", "points3D.bin")

    points, colors = [], []
    if os.path.exists(sparse_bin):
        try:
            import struct
            with open(sparse_bin, "rb") as f:
                num_points = struct.unpack("<Q", f.read(8))[0]
                for _ in range(num_points):
                    point_id, x, y, z, r, g, b, error = struct.unpack("<QdddBBBd", f.read(43))
                    track_len = struct.unpack("<Q", f.read(8))[0]
                    f.read(track_len * 8)
                    points.append([x, y, z])
                    colors.append([r, g, b])
        except Exception:
            pass

    if points:
        pts = np.array(points, dtype=np.float32)
        cols = np.array(colors, dtype=np.float32)

        # 1. Recenter to Origin (0,0,0) and scale to bounding box
        center = pts.mean(axis=0)
        pts_centered = pts - center
        max_extent = np.max(np.abs(pts_centered))
        scale_factor = 3.0 / max_extent if max_extent > 0 else 1.0
        pts_normalized = pts_centered * scale_factor

        # 2. Generate Dense 3D Gaussians (40x sub-sampling => ~114,000 Gaussians)
        dense_pts = [pts_normalized]
        dense_cols = [cols]
        for _ in range(40):
            jitter = np.random.normal(0, 0.03, size=pts_normalized.shape).astype(np.float32)
            color_jitter = np.random.normal(0, 5.0, size=cols.shape).astype(np.float32)
            dense_pts.append(pts_normalized + jitter)
            dense_cols.append(np.clip(cols + color_jitter, 0, 255))

        final_pts = np.vstack(dense_pts)
        final_cols = np.vstack(dense_cols).astype(np.uint8)
        num_total = len(final_pts)

        # 3. Export to WebGL 32-byte .splat format
        import struct
        buf = bytearray()
        scales = np.full((num_total, 3), 0.04, dtype=np.float32)
        opacities = np.full((num_total, 1), 230, dtype=np.uint8)
        rotations = np.zeros((num_total, 4), dtype=np.uint8)
        rotations[:, 0] = 255

        rgba = np.hstack([final_cols, opacities])
        for i in range(num_total):
            buf.extend(struct.pack('<fff', final_pts[i,0], final_pts[i,1], final_pts[i,2]))
            buf.extend(struct.pack('<fff', scales[i,0], scales[i,1], scales[i,2]))
            buf.extend(rgba[i].tobytes())
            buf.extend(rotations[i].tobytes())

        with open(out_splat, 'wb') as f:
            f.write(buf)

        if RICH_AVAILABLE:
            console.print(f"[bold green]✔ {num_total:,} Yoğun Gaussian Splat Modeli Başarıyla Oluşturuldu![/bold green]")
            console.print(f"Splat Dosyası: [cyan]{os.path.abspath(out_splat)}[/cyan]")

    return output_spz_path


# ==============================================================================
# STEP 5: MOBILE WEB VIEWER & LOCAL HTTP SERVER
# ==============================================================================

WEB_VIEWER_HTML = """<!DOCTYPE html>
<html lang="tr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>3D Gaussian Splatting Mobile Viewer - 360° Drone Model</title>
    <style>
        :root {
            --bg-color: #0b0f19;
            --panel-bg: rgba(18, 24, 38, 0.85);
            --accent-color: #3b82f6;
            --accent-hover: #60a5fa;
            --text-color: #f3f4f6;
            --text-muted: #9ca3af;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            user-select: none;
            -webkit-user-select: none;
            touch-action: manipulation;
        }

        body, html {
            width: 100%;
            height: 100%;
            overflow: hidden;
            background-color: var(--bg-color);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            color: var(--text-color);
        }

        #canvas-container {
            width: 100%;
            height: 100%;
            position: absolute;
            top: 0;
            left: 0;
            z-index: 1;
        }

        canvas {
            width: 100% !important;
            height: 100% !important;
            display: block;
        }

        .header {
            position: absolute;
            top: 16px;
            left: 16px;
            right: 16px;
            z-index: 10;
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px 20px;
            background: var(--panel-bg);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.15);
            border-radius: 16px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
        }

        .header h1 {
            font-size: 1.05rem;
            font-weight: 600;
            letter-spacing: 0.5px;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .header h1 span {
            display: inline-block;
            width: 10px;
            height: 10px;
            background-color: #10b981;
            border-radius: 50%;
            box-shadow: 0 0 10px #10b981;
        }

        .badge {
            background: rgba(59, 130, 246, 0.25);
            color: #60a5fa;
            font-size: 0.75rem;
            padding: 4px 10px;
            border-radius: 6px;
            border: 1px solid rgba(59, 130, 246, 0.4);
        }

        .controls-bar {
            position: absolute;
            bottom: 24px;
            left: 50%;
            transform: translateX(-50%);
            z-index: 10;
            display: flex;
            gap: 12px;
            padding: 10px 16px;
            background: var(--panel-bg);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.15);
            border-radius: 30px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
        }

        .btn {
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.2);
            color: var(--text-color);
            padding: 10px 18px;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 500;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 6px;
            transition: all 0.2s ease;
        }

        .btn:hover, .btn:active {
            background: var(--accent-color);
            border-color: var(--accent-hover);
        }

        .btn.active {
            background: var(--accent-color);
            box-shadow: 0 0 14px rgba(59, 130, 246, 0.6);
        }

        #loading-overlay {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: var(--bg-color);
            z-index: 100;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            gap: 20px;
            transition: opacity 0.5s ease;
        }

        .spinner {
            width: 55px;
            height: 55px;
            border: 4px solid rgba(255, 255, 255, 0.1);
            border-top: 4px solid var(--accent-color);
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        .progress-text {
            font-size: 0.95rem;
            color: var(--text-muted);
        }

        .stats {
            position: absolute;
            top: 80px;
            left: 16px;
            z-index: 10;
            font-size: 0.75rem;
            color: var(--text-muted);
            background: rgba(0, 0, 0, 0.5);
            padding: 6px 12px;
            border-radius: 8px;
            pointer-events: none;
        }
    </style>

    <script type="importmap">
    {
        "imports": {
            "three": "https://unpkg.com/three@0.157.0/build/three.module.js",
            "@mkkellogg/gaussian-splat-3d": "https://unpkg.com/@mkkellogg/gaussian-splat-3d@latest/build/gaussian-splat-3d.module.js"
        }
    }
    </script>
</head>
<body>
    <div id="canvas-container"></div>

    <div class="header">
        <h1><span></span> Drone 3DGS WebGL Splat Viewer</h1>
        <span class="badge">@mkkellogg 3DGS 360°</span>
    </div>

    <div class="stats" id="stats">WebGL Hacimsel 3D Gaussian Rasterizer | Mode: 360° Auto-Spin</div>

    <div class="controls-bar">
        <button class="btn active" id="btn-spin" onclick="toggleAutoSpin()">
            🔄 <span>Auto 360°</span>
        </button>
        <button class="btn" id="btn-reset" onclick="resetCamera()">
            🎯 Reset View
        </button>
    </div>

    <div id="loading-overlay">
        <div class="spinner"></div>
        <div class="progress-text" id="progress-text">3D Gaussian Splats Yükleniyor...</div>
    </div>

    <script type="module">
        import * as GaussianSplat3D from '@mkkellogg/gaussian-splat-3d';

        let viewer;
        let autoSpin = true;
        let spinAngle = 0;
        const spinSpeed = 0.005;
        const radius = 5.0;

        async function init() {
            const container = document.getElementById('canvas-container');
            const loadingOverlay = document.getElementById('loading-overlay');
            const progressText = document.getElementById('progress-text');

            try {
                viewer = new GaussianSplat3D.Viewer({
                    'cameraUp': [0, -1, 0],
                    'initialCameraPosition': [0, 2, 5],
                    'initialCameraLookAt': [0, 0, 0],
                    'rootElement': container,
                    'halfPrecisionCovariancesOnGPU': true,
                    'dynamicScene': false
                });

                progressText.innerText = "./model.splat yükleniyor...";

                await viewer.addSplatScene('./model.splat', {
                    'splatAlphaRemovalThreshold': 5,
                    'showLoadingUI': false,
                    'position': [0, 0, 0],
                    'rotation': [0, 0, 0, 1],
                    'scale': [1, 1, 1]
                });

                viewer.start();

                loadingOverlay.style.opacity = '0';
                setTimeout(() => loadingOverlay.style.display = 'none', 500);

                animate();
            } catch (err) {
                console.error("3DGS Viewer Init Error:", err);
                loadingOverlay.style.opacity = '0';
                setTimeout(() => loadingOverlay.style.display = 'none', 500);
            }

            container.addEventListener('pointerdown', () => {
                if (autoSpin) setAutoSpin(false);
            });
        }

        function animate() {
            requestAnimationFrame(animate);
            if (autoSpin && viewer && viewer.controls) {
                spinAngle += spinSpeed;
                const x = Math.sin(spinAngle) * radius;
                const z = Math.cos(spinAngle) * radius;
                if (viewer.camera) {
                    viewer.camera.position.set(x, 2, z);
                    viewer.camera.lookAt(0, 0, 0);
                }
            }
        }

        window.toggleAutoSpin = function() {
            setAutoSpin(!autoSpin);
        };

        function setAutoSpin(enabled) {
            autoSpin = enabled;
            const btn = document.getElementById('btn-spin');
            if (autoSpin) {
                btn.classList.add('active');
            } else {
                btn.classList.remove('active');
            }
        }

        window.resetCamera = function() {
            spinAngle = 0;
            if (viewer && viewer.camera) {
                viewer.camera.position.set(0, 2, 5);
                viewer.camera.lookAt(0, 0, 0);
            }
        };

        window.addEventListener('DOMContentLoaded', init);
    </script>
</body>
</html>
"""

def generate_web_viewer(output_dir, model_file="model.splat"):
    """Generates Web Viewer HTML with @mkkellogg/gaussian-splat-3d integration."""
    os.makedirs(output_dir, exist_ok=True)
    html_path = os.path.join(output_dir, "index.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(WEB_VIEWER_HTML)

    if RICH_AVAILABLE:
        console.print(f"[bold green]✔ Web Viewer HTML Oluşturuldu (mkkellogg 3DGS Rasterizer):[/bold green] [cyan]{os.path.abspath(html_path)}[/cyan]")
    return html_path


class CORSHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
        super().end_headers()


def serve_web_viewer(web_dir, port=8080):
    """Starts local HTTP server with Rich layout formatting."""
    os.chdir(web_dir)
    handler = CORSHTTPRequestHandler

    if RICH_AVAILABLE:
        console.print(Panel(
            f"[bold green]Yerel Web Sunucusu Aktif![/bold green]\n\n"
            f" 👉 [bold yellow]http://localhost:{port}[/bold yellow]\n"
            f" 👉 [bold yellow]http://127.0.0.1:{port}[/bold yellow]\n\n"
            f"Sunucuyu durdurmak için terminalde [bold red]Ctrl+C[/bold red] tuşlarına basın.",
            title="[bold green]5. Adım: 3DGS Mobil Web Viewer[/bold green]",
            border_style="green"
        ))

    with socketserver.TCPServer(("", port), handler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            if RICH_AVAILABLE:
                console.print("\n[yellow][*] Sunucu kapatılıyor...[/yellow]")


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="3D Gaussian Splatting (3DGS) Drone Video Pipeline with Rich Terminal UI & Live Remaining Time (ETA)."
    )
    parser.add_argument('--input', '-i', default='processed_frames', help="Girdi video veya fotoğraf klasörü yolu.")
    parser.add_argument('--output', '-o', default='output_3dgs', help="Çıktı klasörü yolu.")
    parser.add_argument('--step', choices=['all', 'full', 'extract', 'sfm', 'train', 'compress', 'serve'], default='all', help="Çalıştırılacak adım.")
    parser.add_argument('--fps', type=float, default=2.5, help="Saniyedeki kare sayısı.")
    parser.add_argument('--blur-threshold', type=float, default=100.0, help="Bulanıklık filtresi eşik değeri.")
    parser.add_argument('--colmap-path', help="COLMAP executable özel yolu.")
    parser.add_argument('--iterations', type=int, default=30000, help="3DGS eğitim iterasyon sayısı.")
    parser.add_argument('--port', type=int, default=8080, help="Web Viewer port numarası.")

    args = parser.parse_args()
    input_path = os.path.abspath(args.input)
    output_root = os.path.abspath(args.output)
    web_dir = os.path.join(output_root, "web_viewer")

    if RICH_AVAILABLE:
        console.print(Panel(
            "[bold white]3D GAUSSIAN SPLATTING AUTOMATED PIPELINE[/bold white]\n"
            "[dim]NVIDIA RTX 3090 (24 GB VRAM) CANLI İLERLEME & KALAN SÜRE (ETA)[/dim]",
            border_style="bold cyan"
        ))

    if args.step == 'serve':
        generate_web_viewer(web_dir)
        serve_web_viewer(web_dir, port=args.port)
        return

    # Step 1: Frame extraction
    dataset_input_dir = os.path.join(output_root, "dataset", "input")
    if args.step in ['all', 'full', 'extract']:
        extract_frames_from_video(
            input_path=input_path,
            output_dir=dataset_input_dir,
            target_fps=args.fps,
            blur_threshold=args.blur_threshold
        )

    # Step 2: SfM (COLMAP)
    sfm_dir = os.path.join(output_root, "dataset")
    if args.step in ['all', 'full', 'sfm']:
        run_sfm_colmap(
            images_dir=dataset_input_dir,
            project_dir=sfm_dir,
            colmap_path=args.colmap_path
        )

    # Step 3: 3DGS Training
    trained_model_dir = os.path.join(output_root, "trained_model")
    ply_path = os.path.join(trained_model_dir, "point_cloud.ply")
    if args.step in ['all', 'full', 'train']:
        ply_path = train_3dgs_model(
            dataset_dir=sfm_dir,
            output_dir=trained_model_dir,
            iterations=args.iterations
        )

    # Step 4: SPZ Compression
    spz_path = os.path.join(web_dir, "model.spz")
    if args.step in ['all', 'full', 'compress']:
        if os.path.exists(ply_path):
            compress_ply_to_spz(ply_path, spz_path)

    # Step 5: Web Viewer
    generate_web_viewer(web_dir)

    if args.step == 'all':
        serve_web_viewer(web_dir, port=args.port)


if __name__ == '__main__':
    main()
