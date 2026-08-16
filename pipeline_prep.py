import os
import sys
import glob
import time
import argparse
import subprocess

# Windows konsolu için UTF-8 kodlama desteği ayarla
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

def ensure_dependencies():
    """Bağımlılıkların kurulu olup olmadığını kontrol eder, eksikleri pip ile kurar."""
    required = {
        'cv2': 'opencv-python',
        'numpy': 'numpy',
        'ffmpeg': 'ffmpeg-python'
    }
    missing = []
    for module, pkg in required.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(pkg)
    
    if missing:
        print(f"[!] Eksik bağımlılıklar tespit edildi: {', '.join(missing)}")
        print("[*] Otomatik kurulum başlatılıyor...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
            print("[+] Tüm kütüphaneler başarıyla kuruldu.\n")
        except Exception as e:
            print(f"[X] Otomatik kurulum sırasında hata oluştu: {e}")
            print("Lütfen aşağıdaki komutu elle çalıştırın:")
            print(f"    pip install {' '.join(missing)}")
            sys.exit(1)

# Bağımlılık kontrolünü yap
ensure_dependencies()

import cv2
import numpy as np


def calculate_blur_score(frame):
    """
    Karedeki netlik/bulanıklık derecesini Laplacian varyansı kullanarak hesaplar.
    Yüksek değer -> Daha net görsel
    Düşük değer -> Bulanık / Odak dışı / Hareket fluşu
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def scan_videos(input_dir, exclude_patterns=None):
    """
    Girdi dizinindeki .mp4, .mov, .avi videolarını tarar.
    Filtreleme kriterlerine göre hariç tutulacakları eler.
    """
    extensions = ['*.mp4', '*.MOV', '*.mov', '*.avi', '*.mkv']
    video_files = []
    for ext in extensions:
        video_files.extend(glob.glob(os.path.join(input_dir, ext)))
    
    video_files = sorted(list(set(video_files)))
    
    if not exclude_patterns:
        return video_files, []

    excluded_files = []
    filtered_videos = []
    
    for idx, vid_path in enumerate(video_files, 1):
        filename = os.path.basename(vid_path)
        should_exclude = False
        
        for pat in exclude_patterns:
            pat_str = str(pat).strip()
            if pat_str.isdigit():
                # Sadece 1-tabanlı sayısal indeks eşleşmesi (örn: 4 veya 5)
                if int(pat_str) == idx:
                    should_exclude = True
                    break
            else:
                # İsim/metin parçası eşleşmesi (örn: panorama, 162225)
                if pat_str.lower() in filename.lower():
                    should_exclude = True
                    break
        
        if should_exclude:
            excluded_files.append(vid_path)
        else:
            filtered_videos.append(vid_path)

    return filtered_videos, excluded_files


def process_videos(video_list, output_dir, target_fps=2.0, blur_threshold=100.0, jpeg_quality=95):
    """
    Videolardan belirlenen FPS oranında kare çıkarır ve Laplacian varyans filtresi uygular.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    stats = {
        'total_videos': len(video_list),
        'total_extracted': 0,
        'saved_frames': 0,
        'blurry_skipped': 0,
        'start_time': time.time()
    }
    
    print(f"\n{'='*70}")
    print(f" [3D GAUSSIAN SPLATTING DATASET VERİ HAZIRLAMA BAŞLATILDI]")
    print(f"{'='*70}")
    print(f" > Hedef FPS            : {target_fps} kare/saniye")
    print(f" > Bulanıklık Eşik Değeri: {blur_threshold} (Laplacian Varyansı)")
    print(f" > Çıktı Klasörü        : {os.path.abspath(output_dir)}")
    print(f" > JPEG Kalitesi        : %{jpeg_quality}")
    print(f"{'='*70}\n")
    
    total_videos = len(video_list)
    
    for vid_idx, video_path in enumerate(video_list, start=1):
        video_name = os.path.basename(video_path)
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"[X] [{vid_idx}/{total_videos}] Video açılamadı: {video_name}")
            continue
            
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = total_frames / original_fps if original_fps > 0 else 0
        
        # Saniyede target_fps kare alacak adım sayısı (frame stride)
        frame_stride = max(1, int(round(original_fps / target_fps))) if original_fps > 0 else 15
        
        vid_extracted = 0
        vid_saved = 0
        vid_blurry = 0
        
        print(f"[*] [{vid_idx}/{total_videos}] İşleniyor: {video_name}")
        print(f"    └─ FPS: {original_fps:.2f} | Süre: {duration_sec:.1f}s | Kare Adımı: {frame_stride}")
        
        frame_idx = 0
        saved_seq = 1
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_idx % frame_stride == 0:
                vid_extracted += 1
                stats['total_extracted'] += 1
                
                # Bulanıklık Kontrolü
                blur_score = calculate_blur_score(frame)
                
                if blur_score < blur_threshold:
                    vid_blurry += 1
                    stats['blurry_skipped'] += 1
                else:
                    # Net kareyi kaydet
                    frame_filename = f"frame_v{vid_idx:02d}_{saved_seq:05d}_var{int(blur_score)}.jpg"
                    out_path = os.path.join(output_dir, frame_filename)
                    cv2.imwrite(out_path, frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
                    vid_saved += 1
                    saved_seq += 1
                    stats['saved_frames'] += 1
            
            frame_idx += 1
            
        cap.release()
        print(f"    └─ [+] {vid_saved} net kare kaydedildi | [-] {vid_blurry} flulaşmış kare elendi.\n")
        
    stats['elapsed_time'] = time.time() - stats['start_time']
    return stats


def print_report_and_guidance(stats, output_dir):
    """İşlem bitiminde istatistik raporu ve Postshot kullanım rehberini basar."""
    print(f"\n{'='*70}")
    print(f" [İŞLEM TAMAMLANDI - ÖZET RAPOR]")
    print(f"{'='*70}")
    print(f"  • İşlenen Toplam Video : {stats['total_videos']}")
    print(f"  • Çıkarılan Toplam Kare: {stats['total_extracted']}")
    print(f"  • Kaydedilen Net Kare  : {stats['saved_frames']}")
    print(f"  • Elenen Bulanık Kare : {stats['blurry_skipped']}")
    print(f"  • Toplam Süre          : {stats['elapsed_time']:.2f} saniye")
    print(f"  • Çıktı Dizini         : {os.path.abspath(output_dir)}")
    print(f"{'='*70}\n")
    
    print(f"POSTSHOT (3D GAUSSIAN SPLATTING) ADIMLARI:")
    print(f" 1. Postshot uygulamasını açın.")
    print(f" 2. Yeni Proje (New Project) oluşturun.")
    print(f" 3. Oluşturulan net fotoğrafların bulunduğu aşağıdaki klasörü Postshot penceresine sürükleyip bırakın:")
    print(f"    👉  {os.path.abspath(output_dir)}")
    print(f" 4. Postshot ekranında 'Import / Match Images' butonuna basarak kamera pozisyonlarını hesaplatın.")
    print(f" 5. 'Train Gaussian Splatting' seçeneğini başlatın ve 3D modelinizi üretin!\n")


def interactive_exclude_selection(video_files):
    """Kullanıcıya interaktif modda videoları listeleyip hariç tutma seçeneği sunar."""
    print("\n[Dizin İçinde Bulunan Drone Videoları]")
    print("-" * 60)
    for idx, v_path in enumerate(video_files, 1):
        cap = cv2.VideoCapture(v_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        dur = frames / fps if fps > 0 else 0
        cap.release()
        print(f" [{idx}] {os.path.basename(v_path)} ({dur:.1f}s, {fps:.0f} FPS)")
    print("-" * 60)
    
    user_input = input("\nHariç tutmak istediğiniz video numaralarını aralarında virgül ile yazın (örn: 4, 5) veya Enter'a basarak hepsini işleyin: ")
    if not user_input.strip():
        return []
    
    return [item.strip() for item in user_input.split(',')]


def main():
    parser = argparse.ArgumentParser(
        description="Drone videolarından 3D Gaussian Splatting (Postshot) için yüksek kaliteli kare çıkarma ve bulanıklık filtresi otomasyonu."
    )
    parser.add_argument('--input', '-i', default='.', help="Videoların bulunduğu giriş klasörü (Varsayılan: Mevcut klasör)")
    parser.add_argument('--output', '-o', default='processed_frames', help="Karelerin kaydedileceği çıktı klasörü (Varsayılan: processed_frames)")
    parser.add_argument('--fps', type=float, default=2.0, help="Saniyede çıkarılacak kare sayısı (FPS) (Varsayılan: 2.0)")
    parser.add_argument('--blur-threshold', '-b', type=float, default=100.0, help="Laplacian varyansı bulanıklık eşik değeri (Varsayılan: 100.0)")
    parser.add_argument('--exclude', '-e', nargs='+', help="Hariç tutulacak video numaraları veya isim filtreleri (Örn: --exclude 4 5 veya --exclude video4)")
    parser.add_argument('--quality', '-q', type=int, default=95, help="Çıktı JPEG kalitesi 1-100 (Varsayılan: 95)")
    parser.add_argument('--interactive', action='store_true', help="İnteraktif video seçim modunu etkinleştirir")

    args = parser.parse_args()
    
    input_dir = os.path.abspath(args.input)
    output_dir = os.path.abspath(os.path.join(input_dir, args.output))
    
    # Tüm videoları tara
    all_videos, _ = scan_videos(input_dir)
    
    if not all_videos:
        print(f"[X] '{input_dir}' dizininde hiçbir desteklenen video (.mp4, .mov, vb.) bulunamadı.")
        sys.exit(1)
        
    exclude_patterns = args.exclude if args.exclude else []
    
    if args.interactive:
        interactive_excludes = interactive_exclude_selection(all_videos)
        exclude_patterns.extend(interactive_excludes)
        
    selected_videos, excluded_videos = scan_videos(input_dir, exclude_patterns)
    
    if excluded_videos:
        print(f"\n[-] HARİÇ TUTULAN VİDEOLAR ({len(excluded_videos)} adet):")
        for ev in excluded_videos:
            print(f"   • {os.path.basename(ev)}")
            
    if not selected_videos:
        print("\n[!] İşlenecek video kalmadı. Program sonlandırılıyor.")
        sys.exit(0)
        
    stats = process_videos(
        video_list=selected_videos,
        output_dir=output_dir,
        target_fps=args.fps,
        blur_threshold=args.blur_threshold,
        jpeg_quality=args.quality
    )
    
    print_report_and_guidance(stats, output_dir)


if __name__ == '__main__':
    main()
