import gradio as gr
import os
import cv2
import numpy as np
import shutil
import subprocess
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from SinglePhoto import FaceSwapper
import argparse

wellcomingMessage = """
    <h1>Face Swapping Suite ⚡ TURBO</h1>
    <p>All-in-one face swapping: single photo, video, multi-source, and multi-destination!</p>
"""

# ─────────────────────────────────────────────
# ГЛОБАЛЬНЫЙ SWAPPER (ОДИН НА GPU)
# ─────────────────────────────────────────────
# ВАЖНО: на GPU держим ОДИН экземпляр FaceSwapper и вызываем его
# последовательно через lock. Создание FaceSwapper() на каждый поток
# (как было раньше) заново грузит buffalo_l + inswapper веса на КАЖДЫЙ
# вызов — это и есть источник 9 сек/кадр, а не отсутствие GPU.
# Параллельные CUDA-сессии из разных потоков тоже часто конфликтуют
# и не ускоряют, а замедляют/ломают инференс.
_swapper_lock = threading.Lock()
_master_swapper = FaceSwapper()   # грузится ОДИН раз при старте приложения

NUM_WORKERS = 4          # используется только для IO (extract/encode), НЕ для инференса
JPEG_QUALITY = 95

def swap_safe(src_path, src_idx, dst_path, dst_idx):
    """Единая точка вызова модели. Serialized через lock — GPU inference не параллелится потоками."""
    with _swapper_lock:
        return _master_swapper.swap_faces(src_path, int(src_idx), dst_path, int(dst_idx))

# ─────────────────────────────────────────────
# УТИЛИТЫ
# ─────────────────────────────────────────────

def save_img(img, path):
    """Быстрое сохранение с максимальным качеством."""
    cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])

def rgb_to_bgr(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

def add_audio_to_video(original_video_path, video_no_audio_path, output_path):
    cmd = [
        "ffmpeg", "-y",
        "-i", video_no_audio_path,
        "-i", original_video_path,
        "-c:v", "copy", "-c:a", "aac",
        "-map", "0:v:0", "-map", "1:a:0?",
        "-shortest", output_path
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True, ""
    except subprocess.CalledProcessError as e:
        return False, str(e)

def count_frames_fast(video_path):
    """Быстрый подсчёт кадров через ffprobe, без полного декодирования."""
    try:
        cmd = [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-count_packets", "-show_entries", "stream=nb_read_packets",
            "-of", "csv=p=0", video_path
        ]
        out = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return int(out.stdout.strip())
    except Exception:
        # fallback: OpenCV estimate
        cap = cv2.VideoCapture(video_path)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return max(n, 1)

def get_fps(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()
    return fps

def extract_frames_fast(video_path, frames_dir):
    """Быстрое извлечение кадров через ffmpeg (в 2-3х быстрее OpenCV)."""
    os.makedirs(frames_dir, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-q:v", "2",
        os.path.join(frames_dir, "frame_%05d.jpg")
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    frames = sorted([
        os.path.join(frames_dir, f)
        for f in os.listdir(frames_dir)
        if f.endswith(".jpg")
    ])
    return frames

def frames_to_video_fast(frames_dir, output_path, fps):
    """Быстрая сборка видео через ffmpeg напрямую из папки."""
    pattern = os.path.join(frames_dir, "swapped_%05d.jpg")
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", pattern,
        "-c:v", "libx264",
        "-preset", "ultrafast",   # максимальная скорость кодирования
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        output_path
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def make_dirs(*paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)

def cleanup(*paths):
    for p in paths:
        try:
            if os.path.isdir(p):
                shutil.rmtree(p)
            elif os.path.isfile(p):
                os.remove(p)
        except Exception:
            pass

# ─────────────────────────────────────────────
# ЯДРО: параллельная обработка кадров
# ─────────────────────────────────────────────

def _process_frame_single(args):
    """Один кадр: одно лицо."""
    idx, frame_path, src_path, src_idx, dst_idx, out_path = args
    try:
        swapped = swap_safe(src_path, src_idx, frame_path, dst_idx)
        cv2.imwrite(out_path, swapped, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    except Exception:
        shutil.copy2(frame_path, out_path)

def _process_frame_all_faces(args):
    """Один кадр: все лица (N проходов подряд)."""
    idx, frame_path, src_path, num_faces, out_path = args
    try:
        img = cv2.imread(frame_path)
        tmp_path = out_path + ".tmp.jpg"
        cv2.imwrite(tmp_path, img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        for face_idx in range(1, int(num_faces) + 1):
            try:
                result = swap_safe(src_path, 1, tmp_path, face_idx)
                cv2.imwrite(tmp_path, result, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            except Exception:
                pass
        shutil.move(tmp_path, out_path)
    except Exception:
        shutil.copy2(frame_path, out_path)

def _process_frame_custom(args):
    """Один кадр: кастомный маппинг лиц."""
    idx, frame_path, src_paths, mapping, out_path = args
    try:
        tmp_path = out_path + ".tmp.jpg"
        shutil.copy2(frame_path, tmp_path)
        for face_idx, src_idx in enumerate(mapping, start=1):
            if src_idx < 1 or src_idx > len(src_paths):
                continue
            try:
                result = swap_safe(src_paths[src_idx - 1], 1, tmp_path, face_idx)
                cv2.imwrite(tmp_path, result, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            except Exception:
                pass
        shutil.move(tmp_path, out_path)
    except Exception:
        shutil.copy2(frame_path, out_path)

def run_parallel(worker_fn, args_list, progress_cb=None, total=None):
    """Запускает worker_fn параллельно на NUM_WORKERS потоках."""
    total = total or len(args_list)
    done = [0]
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as ex:
        futures = {ex.submit(worker_fn, a): i for i, a in enumerate(args_list)}
        for f in as_completed(futures):
            f.result()
            done[0] += 1
            if progress_cb:
                progress_cb(done[0], total)

# ─────────────────────────────────────────────
# ФОТО-ФУНКЦИИ
# ─────────────────────────────────────────────

def swap_single_photo(src_img, src_idx, dst_img, dst_idx, progress=gr.Progress()):
    log = ""
    t = time.time()
    try:
        progress(0, desc="Preparing")
        make_dirs("SinglePhoto")
        src_path = "SinglePhoto/data_src.jpg"
        dst_path = "SinglePhoto/data_dst.jpg"
        out_path = "SinglePhoto/output_swapped.jpg"
        save_img(rgb_to_bgr(src_img), src_path)
        save_img(rgb_to_bgr(dst_img), dst_path)
        progress(0.4, desc="Swapping")
        result = swap_safe(src_path, int(src_idx), dst_path, int(dst_idx))
        cv2.imwrite(out_path, result)
        cleanup(src_path, dst_path)
        progress(1, desc="Done")
        log += f"Done in {time.time()-t:.2f}s\n"
        return out_path, log
    except Exception as e:
        return None, f"Error: {e}\n"

def swap_single_src_multi_dst(src_img, dst_imgs, dst_indices, progress=gr.Progress()):
    log = ""
    t = time.time()
    make_dirs("SingleSrcMultiDst/src", "SingleSrcMultiDst/dst", "SingleSrcMultiDst/output")
    src_path = "SingleSrcMultiDst/src/data_src.jpg"
    save_img(rgb_to_bgr(src_img if not isinstance(src_img, tuple) else src_img[0]), src_path)

    indices = [int(x.strip()) for x in str(dst_indices).split(",") if x.strip().isdigit()]
    results = []

    def process(j_img):
        j, dst_img = j_img
        if isinstance(dst_img, tuple): dst_img = dst_img[0]
        dst_path = f"SingleSrcMultiDst/dst/dst_{j}.jpg"
        out_path = f"SingleSrcMultiDst/output/out_{j}.jpg"
        save_img(rgb_to_bgr(dst_img), dst_path)
        idx = indices[j] if j < len(indices) else 1
        try:
            result = swap_safe(src_path, 1, dst_path, idx)
            cv2.imwrite(out_path, result)
            return j, out_path
        except Exception as e:
            return j, None

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as ex:
        futs = {ex.submit(process, (j, img)): j for j, img in enumerate(dst_imgs)}
        ordered = {}
        done = 0
        for f in as_completed(futs):
            j, path = f.result()
            ordered[j] = path
            done += 1
            progress(done / len(dst_imgs), desc=f"{done}/{len(dst_imgs)}")

    results = [ordered[k] for k in sorted(ordered) if ordered[k]]
    log += f"Done {len(results)} images in {time.time()-t:.2f}s\n"
    return results, log

def swap_multi_src_single_dst(src_imgs, dst_img, dst_idx, progress=gr.Progress()):
    log = ""
    t = time.time()
    make_dirs("MultiSrcSingleDst/src", "MultiSrcSingleDst/dst", "MultiSrcSingleDst/output")
    dst_path = "MultiSrcSingleDst/dst/data_dst.jpg"
    if isinstance(dst_img, tuple): dst_img = dst_img[0]
    save_img(rgb_to_bgr(dst_img), dst_path)

    def process(i_img):
        i, src_img = i_img
        if isinstance(src_img, tuple): src_img = src_img[0]
        src_path = f"MultiSrcSingleDst/src/src_{i}.jpg"
        out_path = f"MultiSrcSingleDst/output/out_{i}.jpg"
        save_img(rgb_to_bgr(src_img), src_path)
        try:
            result = swap_safe(src_path, 1, dst_path, int(dst_idx))
            cv2.imwrite(out_path, result)
            return i, out_path
        except Exception as e:
            return i, None

    ordered = {}
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as ex:
        futs = {ex.submit(process, (i, img)): i for i, img in enumerate(src_imgs)}
        done = 0
        for f in as_completed(futs):
            i, path = f.result()
            ordered[i] = path
            done += 1
            progress(done / len(src_imgs), desc=f"{done}/{len(src_imgs)}")

    results = [ordered[k] for k in sorted(ordered) if ordered[k]]
    log += f"Done {len(results)} images in {time.time()-t:.2f}s\n"
    return results, log

def swap_multi_src_multi_dst(src_imgs, dst_imgs, dst_indices, progress=gr.Progress()):
    log = ""
    t = time.time()
    make_dirs("MultiSrcMultiDst/src", "MultiSrcMultiDst/dst", "MultiSrcMultiDst/output")
    indices = [int(x.strip()) for x in str(dst_indices).split(",") if x.strip().isdigit()]

    src_paths = []
    for i, src in enumerate(src_imgs):
        if isinstance(src, tuple): src = src[0]
        p = f"MultiSrcMultiDst/src/src_{i}.jpg"
        save_img(rgb_to_bgr(src), p)
        src_paths.append(p)

    tasks = []
    for i, src_path in enumerate(src_paths):
        for j, dst in enumerate(dst_imgs):
            if isinstance(dst, tuple): dst = dst[0]
            dst_path = f"MultiSrcMultiDst/dst/dst_{j}.jpg"
            save_img(rgb_to_bgr(dst), dst_path)
            out_path = f"MultiSrcMultiDst/output/out_{i}_{j}.jpg"
            idx = indices[j] if j < len(indices) else 1
            tasks.append((i, j, src_path, dst_path, out_path, idx))

    results_map = {}
    done = [0]

    def process(task):
        i, j, src_path, dst_path, out_path, idx = task
        try:
            result = swap_safe(src_path, 1, dst_path, idx)
            cv2.imwrite(out_path, result)
            return (i, j), out_path
        except:
            return (i, j), None

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as ex:
        futs = {ex.submit(process, t): t for t in tasks}
        for f in as_completed(futs):
            key, path = f.result()
            results_map[key] = path
            done[0] += 1
            progress(done[0] / len(tasks), desc=f"{done[0]}/{len(tasks)}")

    results = [results_map[k] for k in sorted(results_map) if results_map[k]]
    log += f"Done {len(results)} pairs in {time.time()-t:.2f}s\n"
    return results, log

def swap_faces_custom(src_imgs, dst_img, mapping_str, progress=gr.Progress()):
    log = ""
    t = time.time()
    make_dirs("CustomSwap/src", "CustomSwap")
    dst_path = "CustomSwap/data_dst.jpg"
    out_path = "CustomSwap/output_swapped.jpg"
    save_img(rgb_to_bgr(dst_img), dst_path)

    src_paths = []
    for i, src in enumerate(src_imgs):
        if isinstance(src, tuple): src = src[0]
        if src is None: continue
        p = f"CustomSwap/src/src_{i+1}.jpg"
        save_img(rgb_to_bgr(src), p)
        src_paths.append(p)

    mapping = [int(x.strip()) for x in mapping_str.split(",") if x.strip().isdigit()]
    tmp = dst_path + ".tmp.jpg"
    shutil.copy2(dst_path, tmp)
    total = len(mapping)
    for step, (face_idx, src_idx) in enumerate(enumerate(mapping, start=1)):
        if src_idx < 1 or src_idx > len(src_paths): continue
        try:
            r = swap_safe(src_paths[src_idx - 1], 1, tmp, face_idx)
            cv2.imwrite(tmp, r)
        except Exception as e:
            log += f"Face {face_idx} error: {e}\n"
        progress((step) / total, desc=f"Face {step}/{total}")

    shutil.move(tmp, out_path)
    log += f"Done in {time.time()-t:.2f}s\n"
    return out_path, log

# ─────────────────────────────────────────────
# ВИДЕО-ФУНКЦИИ (турбо)
# ─────────────────────────────────────────────

def _video_pipeline(
    src_path, video_input, swapped_dir, frames_dir,
    output_video_path, final_output_path,
    frame_worker_fn, frame_args_builder,
    add_audio, delete_frames_dir, copy_to_drive,
    progress, log
):
    """Общий пайплайн для всех видео-режимов."""
    t = time.time()

    # Копируем видео
    dst_video_path = os.path.join(os.path.dirname(swapped_dir), "src.mp4")
    if isinstance(video_input, str):
        shutil.copy2(video_input, dst_video_path)
    else:
        shutil.copy2(video_input.name, dst_video_path)

    fps = get_fps(dst_video_path)

    # Извлекаем кадры быстро (ffmpeg)
    progress(0.05, desc="Extracting frames (ffmpeg)...")
    frame_paths = extract_frames_fast(dst_video_path, frames_dir)
    total = len(frame_paths)
    log += f"Extracted {total} frames | FPS={fps:.2f}\n"

    # Строим список аргументов для каждого кадра
    args_list = frame_args_builder(frame_paths, swapped_dir)

    # Параллельная обработка
    done_count = [0]
    start_t = [time.time()]

    def on_progress(done, total):
        done_count[0] = done
        elapsed = time.time() - start_t[0]
        avg = elapsed / done if done else 0
        remaining = avg * (total - done)
        m, s = divmod(int(remaining), 60)
        progress(
            0.1 + 0.7 * done / total,
            desc=f"Swapping {done}/{total} | ETA {m:02d}:{s:02d}"
        )

    progress(0.1, desc=f"Swapping {total} frames with {NUM_WORKERS} workers...")
    run_parallel(frame_worker_fn, args_list, on_progress, total)

    # Сборка видео
    progress(0.82, desc="Building video (ffmpeg ultrafast)...")
    frames_to_video_fast(swapped_dir, output_video_path, fps)
    log += f"Video assembled\n"

    # Google Drive
    if copy_to_drive:
        drive_path = "/content/drive/MyDrive/" + os.path.basename(output_video_path)
        try:
            shutil.copy2(output_video_path, drive_path)
            log += f"Copied to Drive: {drive_path}\n"
        except Exception as e:
            log += f"Drive copy failed: {e}\n"

    # Аудио
    progress(0.9, desc="Adding audio...")
    if add_audio:
        ok, err = add_audio_to_video(dst_video_path, output_video_path, final_output_path)
        if ok:
            log += "Audio added\n"
        else:
            log += f"Audio failed: {err}\n"
            final_output_path = output_video_path
    else:
        final_output_path = output_video_path

    # Очистка
    if delete_frames_dir:
        cleanup(frames_dir, swapped_dir)
    cleanup(dst_video_path, output_video_path if add_audio else None)

    elapsed = time.time() - t
    fps_processed = total / elapsed if elapsed > 0 else 0
    log += f"Total: {elapsed:.1f}s | Speed: {fps_processed:.1f} frames/sec\n"
    progress(1, desc=f"Done! {elapsed:.1f}s")
    return final_output_path, log

# --- Video Single Face ---
def swap_video(src_img, src_idx, video, dst_idx, delete_frames_dir=True, add_audio=True, copy_to_drive=False, progress=gr.Progress()):
    log = ""
    base = "VideoSwapping"
    swapped_dir = f"{base}/swapped_frames"
    frames_dir = f"{base}/video_frames"
    make_dirs(base, swapped_dir, frames_dir)
    src_path = f"{base}/data_src.jpg"
    save_img(rgb_to_bgr(src_img), src_path)

    def builder(frame_paths, swapped_dir):
        return [
            (i, fp, src_path, int(src_idx), int(dst_idx),
             os.path.join(swapped_dir, f"swapped_{i:05d}.jpg"))
            for i, fp in enumerate(frame_paths)
        ]

    return _video_pipeline(
        src_path, video, swapped_dir, frames_dir,
        f"{base}/output_no_audio.mp4", f"{base}/output_with_audio.mp4",
        _process_frame_single, builder,
        add_audio, delete_frames_dir, copy_to_drive, progress, log
    )

# --- Video All Faces ---
def swap_video_all_faces(src_img, video, num_faces_to_swap, delete_frames_dir=True, add_audio=True, copy_to_drive=False, progress=gr.Progress()):
    log = ""
    base = "VideoSwappingAllFaces"
    swapped_dir = f"{base}/swapped_frames"
    frames_dir = f"{base}/video_frames"
    make_dirs(base, swapped_dir, frames_dir)
    src_path = f"{base}/data_src.jpg"
    save_img(rgb_to_bgr(src_img), src_path)

    def builder(frame_paths, swapped_dir):
        return [
            (i, fp, src_path, int(num_faces_to_swap),
             os.path.join(swapped_dir, f"swapped_{i:05d}.jpg"))
            for i, fp in enumerate(frame_paths)
        ]

    return _video_pipeline(
        src_path, video, swapped_dir, frames_dir,
        f"{base}/output_no_audio.mp4", f"{base}/output_with_audio.mp4",
        _process_frame_all_faces, builder,
        add_audio, delete_frames_dir, copy_to_drive, progress, log
    )

# --- Video Custom Mapping ---
def swap_video_custom_mapping(src_imgs, video, mapping_str, delete_frames_dir=True, add_audio=True, copy_to_drive=False, progress=gr.Progress()):
    log = ""
    base = "CustomVideoSwap"
    swapped_dir = f"{base}/swapped_frames"
    frames_dir = f"{base}/frames"
    make_dirs(base, swapped_dir, frames_dir, f"{base}/src")

    src_paths = []
    for i, src in enumerate(src_imgs):
        if isinstance(src, tuple): src = src[0]
        if src is None: continue
        p = f"{base}/src/src_{i+1}.jpg"
        if isinstance(src, np.ndarray):
            save_img(rgb_to_bgr(src), p)
        elif isinstance(src, str) and os.path.exists(src):
            shutil.copy2(src, p)
        src_paths.append(p)

    mapping = [int(x.strip()) for x in mapping_str.split(",") if x.strip().isdigit()]

    def builder(frame_paths, swapped_dir):
        return [
            (i, fp, src_paths, mapping,
             os.path.join(swapped_dir, f"swapped_{i:05d}.jpg"))
            for i, fp in enumerate(frame_paths)
        ]

    return _video_pipeline(
        src_paths[0] if src_paths else "", video, swapped_dir, frames_dir,
        f"{base}/output_no_audio.mp4", f"{base}/output_with_audio.mp4",
        _process_frame_custom, builder,
        add_audio, delete_frames_dir, copy_to_drive, progress, log
    )

# ─────────────────────────────────────────────
# BATCH-ВИДЕО (параллельные видео)
# ─────────────────────────────────────────────

def _process_one_video(args):
    """Обрабатывает одно видео из батча. Отчитывается о каждом кадре через shared_counter."""
    (
        video_path, src_path, frame_worker_fn, frame_args_builder_fn,
        base_dir, i, add_audio, delete_frames_dir, copy_to_drive,
        shared_counter, counter_lock, on_frame_done
    ) = args

    vid_name = os.path.basename(video_path) if isinstance(video_path, str) else f"vid_{i}.mp4"
    curr_dir = os.path.join(base_dir, f"vid_{i}")
    frames_dir = os.path.join(curr_dir, "frames")
    swapped_dir = os.path.join(curr_dir, "swapped")
    local_copy = os.path.join(curr_dir, "src.mp4")
    out_no_audio = os.path.join(curr_dir, "out_no_audio.mp4")
    out_final = os.path.join(curr_dir, f"swapped_{vid_name}")
    make_dirs(curr_dir, frames_dir, swapped_dir)

    src = video_path if isinstance(video_path, str) else video_path.name
    shutil.copy2(src, local_copy)
    fps = get_fps(local_copy)
    frame_paths = extract_frames_fast(local_copy, frames_dir)

    args_list = frame_args_builder_fn(frame_paths, swapped_dir, src_path, i)

    def _wrapped(a):
        frame_worker_fn(a)
        with counter_lock:
            shared_counter[0] += 1
            on_frame_done(shared_counter[0])

    # Внутри батча используем меньше воркеров чтобы не конкурировать
    inner_workers = max(1, NUM_WORKERS // 2)
    with ThreadPoolExecutor(max_workers=inner_workers) as ex:
        list(ex.map(_wrapped, args_list))

    frames_to_video_fast(swapped_dir, out_no_audio, fps)

    if add_audio:
        ok, _ = add_audio_to_video(local_copy, out_no_audio, out_final)
        if not ok:
            out_final = out_no_audio

    if copy_to_drive:
        drive_path = "/content/drive/MyDrive/" + os.path.basename(out_final)
        try:
            shutil.copy2(out_final, drive_path)
        except Exception:
            pass

    result_path = out_final
    # Копируем финал наружу перед удалением папки
    final_copy = os.path.join(base_dir, f"result_{i}_{vid_name}")
    shutil.copy2(result_path, final_copy)

    if delete_frames_dir:
        cleanup(curr_dir)

    return final_copy

def swap_single_src_multi_video(src_img, dst_videos, dst_indices, delete_frames_dir=True, add_audio=True, copy_to_drive=False, progress=gr.Progress(track_tqdm=True)):
    log = ""
    t = time.time()
    base = "SingleSrcMultiVideo"
    make_dirs(base)
    src_path = os.path.join(base, "data_src.jpg")
    save_img(rgb_to_bgr(src_img), src_path)

    indices = [int(x.strip()) for x in str(dst_indices).split(",") if x.strip().isdigit()]

    def frame_args_builder(frame_paths, swapped_dir, src_path, vid_i):
        dst_idx = indices[vid_i] if vid_i < len(indices) else 1
        return [
            (i, fp, src_path, 1, dst_idx,
             os.path.join(swapped_dir, f"swapped_{i:05d}.jpg"))
            for i, fp in enumerate(frame_paths)
        ]

    # Считаем общее число кадров заранее, чтобы прогресс был честным (по кадрам, не по видео)
    progress(0, desc="Counting frames...")
    video_paths_resolved = [v if isinstance(v, str) else v.name for v in dst_videos]
    total_frames = sum(count_frames_fast(vp) for vp in video_paths_resolved)
    total_frames = max(total_frames, 1)

    shared_counter = [0]
    counter_lock = threading.Lock()
    start_t = time.time()

    def on_frame_done(done_frames):
        elapsed = time.time() - start_t
        avg = elapsed / done_frames if done_frames else 0
        remaining = avg * (total_frames - done_frames)
        m, s = divmod(int(remaining), 60)
        progress(
            min(done_frames / total_frames, 1.0),
            desc=f"Frames {done_frames}/{total_frames} | ETA {m:02d}:{s:02d}"
        )

    batch_args = [
        (v, src_path, _process_frame_single, frame_args_builder,
         base, i, add_audio, delete_frames_dir, copy_to_drive,
         shared_counter, counter_lock, on_frame_done)
        for i, v in enumerate(dst_videos)
    ]

    results = []
    done_videos = [0]
    with ThreadPoolExecutor(max_workers=2) as ex:  # 2 видео параллельно
        futs = {ex.submit(_process_one_video, a): i for i, a in enumerate(batch_args)}
        for f in as_completed(futs):
            path = f.result()
            results.append(path)
            done_videos[0] += 1

    progress(1.0, desc="Done")
    log += f"Done {len(results)} videos in {time.time()-t:.1f}s\n"
    return sorted(results), log

def swap_single_src_multi_video_simple(src_img, src_idx, dst_videos, dst_idx, delete_frames_dir=True, add_audio=True, copy_to_drive=False, progress=gr.Progress(track_tqdm=True)):
    indices_str = ",".join([str(int(dst_idx))] * len(dst_videos))
    return swap_single_src_multi_video(src_img, dst_videos, indices_str, delete_frames_dir, add_audio, copy_to_drive, progress)

def swap_batch_all_faces(src_img, video_files, num_faces_to_swap, delete_frames_dir=True, add_audio=True, copy_to_drive=False, progress=gr.Progress()):
    log = ""
    t = time.time()
    base = "BatchAllFaces"
    make_dirs(base)
    src_path = os.path.join(base, "data_src.jpg")
    save_img(rgb_to_bgr(src_img), src_path)

    def frame_args_builder(frame_paths, swapped_dir, src_path, vid_i):
        return [
            (i, fp, src_path, int(num_faces_to_swap),
             os.path.join(swapped_dir, f"swapped_{i:05d}.jpg"))
            for i, fp in enumerate(frame_paths)
        ]

    progress(0, desc="Counting frames...")
    video_paths_resolved = [v if isinstance(v, str) else v.name for v in video_files]
    total_frames = max(sum(count_frames_fast(vp) for vp in video_paths_resolved), 1)

    shared_counter = [0]
    counter_lock = threading.Lock()
    start_t = time.time()

    def on_frame_done(done_frames):
        elapsed = time.time() - start_t
        avg = elapsed / done_frames if done_frames else 0
        remaining = avg * (total_frames - done_frames)
        m, s = divmod(int(remaining), 60)
        progress(
            min(done_frames / total_frames, 1.0),
            desc=f"Frames {done_frames}/{total_frames} | ETA {m:02d}:{s:02d}"
        )

    batch_args = [
        (v, src_path, _process_frame_all_faces, frame_args_builder,
         base, i, add_audio, delete_frames_dir, copy_to_drive,
         shared_counter, counter_lock, on_frame_done)
        for i, v in enumerate(video_files)
    ]

    results = []
    with ThreadPoolExecutor(max_workers=2) as ex:
        futs = {ex.submit(_process_one_video, a): i for i, a in enumerate(batch_args)}
        for f in as_completed(futs):
            results.append(f.result())

    progress(1.0, desc="Done")
    log += f"Done {len(results)} videos in {time.time()-t:.1f}s\n"
    return sorted(results), log

def swap_batch_custom(src_imgs, video_files, mapping_str, delete_frames_dir=True, add_audio=True, copy_to_drive=False, progress=gr.Progress()):
    log = ""
    t = time.time()
    base = "BatchCustom"
    src_store = os.path.join(base, "sources")
    make_dirs(base, src_store)

    src_paths = []
    for k, src in enumerate(src_imgs):
        if isinstance(src, tuple): src = src[0]
        p = os.path.join(src_store, f"src_{k+1}.jpg")
        if isinstance(src, np.ndarray):
            save_img(rgb_to_bgr(src), p)
            src_paths.append(p)
        elif isinstance(src, str) and os.path.exists(src):
            shutil.copy2(src, p)
            src_paths.append(p)

    mapping = [int(x.strip()) for x in mapping_str.split(",") if x.strip().isdigit()]

    def frame_args_builder(frame_paths, swapped_dir, _src_path, vid_i):
        return [
            (i, fp, src_paths, mapping,
             os.path.join(swapped_dir, f"swapped_{i:05d}.jpg"))
            for i, fp in enumerate(frame_paths)
        ]

    progress(0, desc="Counting frames...")
    video_paths_resolved = [v if isinstance(v, str) else v.name for v in video_files]
    total_frames = max(sum(count_frames_fast(vp) for vp in video_paths_resolved), 1)

    shared_counter = [0]
    counter_lock = threading.Lock()
    start_t = time.time()

    def on_frame_done(done_frames):
        elapsed = time.time() - start_t
        avg = elapsed / done_frames if done_frames else 0
        remaining = avg * (total_frames - done_frames)
        m, s = divmod(int(remaining), 60)
        progress(
            min(done_frames / total_frames, 1.0),
            desc=f"Frames {done_frames}/{total_frames} | ETA {m:02d}:{s:02d}"
        )

    batch_args = [
        (v, src_paths[0] if src_paths else "", _process_frame_custom, frame_args_builder,
         base, i, add_audio, delete_frames_dir, copy_to_drive,
         shared_counter, counter_lock, on_frame_done)
        for i, v in enumerate(video_files)
    ]

    results = []
    with ThreadPoolExecutor(max_workers=2) as ex:
        futs = {ex.submit(_process_one_video, a): i for i, a in enumerate(batch_args)}
        for f in as_completed(futs):
            results.append(f.result())

    progress(1.0, desc="Done")
    log += f"Done {len(results)} videos in {time.time()-t:.1f}s\n"
    return sorted(results), log

# ─────────────────────────────────────────────
# GRADIO UI
# ─────────────────────────────────────────────

with gr.Blocks() as demo:
    gr.Markdown(wellcomingMessage)

    with gr.Tab("Single Photo"):
        gr.Interface(fn=swap_single_photo, inputs=[
            gr.Image(label="Source Image"), gr.Number(value=1, label="Source Face Index"),
            gr.Image(label="Destination Image"), gr.Number(value=1, label="Destination Face Index"),
        ], outputs=[gr.Image(label="Result"), gr.Textbox(label="Log", lines=5)])

    with gr.Tab("SingleSrc MultiDst (Photos)"):
        gr.Interface(fn=swap_single_src_multi_dst, inputs=[
            gr.Image(label="Source Image"),
            gr.Gallery(label="Destination Images", type="numpy", columns=3),
            gr.Textbox(label="Destination Face Indices (e.g. 1,1,2)"),
        ], outputs=[gr.Gallery(label="Results"), gr.Textbox(label="Log", lines=5)])

    with gr.Tab("MultiSrc SingleDst (Photos)"):
        gr.Interface(fn=swap_multi_src_single_dst, inputs=[
            gr.Gallery(label="Source Images", type="numpy", columns=3),
            gr.Image(label="Destination Image"),
            gr.Number(value=1, label="Destination Face Index"),
        ], outputs=[gr.Gallery(label="Results"), gr.Textbox(label="Log", lines=5)])

    with gr.Tab("MultiSrc MultiDst (Photos)"):
        gr.Interface(fn=swap_multi_src_multi_dst, inputs=[
            gr.Gallery(label="Source Images", type="numpy", columns=3),
            gr.Gallery(label="Destination Images", type="numpy", columns=3),
            gr.Textbox(label="Destination Face Indices (e.g. 1,1,2)"),
        ], outputs=[gr.Gallery(label="Results"), gr.Textbox(label="Log", lines=5)])

    with gr.Tab("Custom Face Mapping (Photos)"):
        gr.Interface(fn=swap_faces_custom, inputs=[
            gr.Gallery(label="Source Images", type="numpy", columns=3),
            gr.Image(label="Destination Image"),
            gr.Textbox(label="Mapping (e.g. 2,1,3)"),
        ], outputs=[gr.Image(label="Result"), gr.Textbox(label="Log", lines=5)])

    with gr.Tab("Video Swapping (Single)"):
        gr.Interface(fn=swap_video, inputs=[
            gr.Image(label="Source Image"), gr.Number(value=1, label="Source Face Index"),
            gr.Video(label="Target Video"), gr.Number(value=1, label="Destination Face Index"),
            gr.Checkbox(label="Delete frames dir", value=True),
            gr.Checkbox(label="Add audio", value=True),
            gr.Checkbox(label="Copy to Drive", value=False),
        ], outputs=[gr.Video(label="Result"), gr.Textbox(label="Log", lines=5)])

    with gr.Tab("Video All Faces"):
        gr.Interface(fn=swap_video_all_faces, inputs=[
            gr.Image(label="Source Image"),
            gr.Video(label="Target Video"),
            gr.Number(value=1, label="Number of Faces to Swap"),
            gr.Checkbox(label="Delete frames dir", value=True),
            gr.Checkbox(label="Add audio", value=True),
            gr.Checkbox(label="Copy to Drive", value=False),
        ], outputs=[gr.Video(label="Result"), gr.Textbox(label="Log", lines=5)])

    with gr.Tab("Custom Video Mapping"):
        gr.Interface(fn=swap_video_custom_mapping, inputs=[
            gr.Gallery(label="Source Images", type="numpy", columns=3),
            gr.Video(label="Target Video"),
            gr.Textbox(label="Mapping (e.g. 2,1,3)"),
            gr.Checkbox(label="Delete frames dir", value=True),
            gr.Checkbox(label="Add audio", value=True),
            gr.Checkbox(label="Copy to Drive", value=False),
        ], outputs=[gr.Video(label="Result"), gr.Textbox(label="Log", lines=5)])

    with gr.Tab("Batch Video (Specific Face)"):
        gr.Markdown("### Swap one face into specific target face across MULTIPLE videos.")
        gr.Interface(fn=swap_single_src_multi_video, inputs=[
            gr.Image(label="Source Image"),
            gr.File(label="Target Videos", file_count="multiple", type="filepath"),
            gr.Textbox(label="Destination Face Indices per video (e.g. 1,2,1)"),
            gr.Checkbox(label="Delete frames dir", value=True),
            gr.Checkbox(label="Add audio", value=True),
            gr.Checkbox(label="Copy to Drive", value=False),
        ], outputs=[gr.Gallery(label="Results", type="filepath"), gr.Textbox(label="Log", lines=5)])

    with gr.Tab("Batch Video (All Faces)"):
        gr.Markdown("### Swap one face into ALL faces across MULTIPLE videos.")
        gr.Interface(fn=swap_batch_all_faces, inputs=[
            gr.Image(label="Source Image"),
            gr.File(label="Target Videos", file_count="multiple", type="filepath"),
            gr.Number(value=5, label="Max Faces per Frame"),
            gr.Checkbox(label="Delete frames dir", value=True),
            gr.Checkbox(label="Add audio", value=True),
            gr.Checkbox(label="Copy to Drive", value=False),
        ], outputs=[gr.Gallery(label="Results", type="filepath"), gr.Textbox(label="Log", lines=5)])

    with gr.Tab("Batch Video (Custom Mapping)"):
        gr.Markdown("### Multiple sources → specific faces in MULTIPLE videos.")
        gr.Interface(fn=swap_batch_custom, inputs=[
            gr.Gallery(label="Source Images", type="numpy", columns=3),
            gr.File(label="Target Videos", file_count="multiple", type="filepath"),
            gr.Textbox(label="Mapping (e.g. 2,1,3)"),
            gr.Checkbox(label="Delete frames dir", value=True),
            gr.Checkbox(label="Add audio", value=True),
            gr.Checkbox(label="Copy to Drive", value=False),
        ], outputs=[gr.Gallery(label="Results", type="filepath"), gr.Textbox(label="Log", lines=5)])

    with gr.Tab("⚡ БЫСТРЫЙ Multi-Video"):
        gr.Interface(fn=swap_single_src_multi_video_simple, inputs=[
            gr.Image(label="Source Image"),
            gr.Number(value=1, label="Source Face Index"),
            gr.File(label="Target Videos", file_count="multiple", type="filepath"),
            gr.Number(value=1, label="Destination Face Index (для ВСЕХ видео)"),
            gr.Checkbox(label="Delete frames dir", value=True),
            gr.Checkbox(label="Add audio", value=True),
            gr.Checkbox(label="Copy to Drive", value=False),
        ], outputs=[gr.Gallery(label="Результаты", type="filepath"), gr.Textbox(label="Log", lines=5)])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel frame workers")
    args = parser.parse_args()
    NUM_WORKERS = args.workers
    print(f"⚡ Starting with {NUM_WORKERS} workers")
    demo.launch(share=args.share)