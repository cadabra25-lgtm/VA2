import tkinter as tk
import tkinter.ttk as ttk
from tkinter import filedialog, messagebox
import threading
import queue
import sys
import os
import time
from datetime import datetime, date
from decord import VideoReader, cpu
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageTk
import glob
import cv2
import subprocess
import platform
import httpx
from lxml import etree
import uuid


# ==============================
# Константы для Hikvision
# ==============================
#USER = "admin"
#PASSWORD = "qweR-123"
USER = "Operator"
PASSWORD = "Pobeda123"

NVR_IPS = ["172.28.10.21", "172.28.10.22", "172.28.10.23", "172.28.10.24", "172.28.10.25"]


# ==============================
# API Functions для Hikvision
# ==============================
def parse_streaming_channels(xml_content):
    try:
        root = etree.fromstring(xml_content)
        for elem in root.iter():
            if elem.tag.startswith('{'):
                elem.tag = elem.tag.split('}', 1)[1]
        camera_numbers = set()
        for channel in root.xpath('.//StreamingChannel'):
            ch_id = channel.findtext('id', 'N/A')
            if ch_id != 'N/A' and ch_id.endswith('1'):
                try:
                    if len(ch_id) == 3:
                        cam_num = int(ch_id[0])
                    elif len(ch_id) == 4:
                        cam_num = int(ch_id[:2])
                    else:
                        continue
                    if 1 <= cam_num <= 32:
                        camera_numbers.add(cam_num)
                except ValueError:
                    continue
        return sorted(camera_numbers)
    except Exception as e:
        print(f"❌ Ошибка парсинга streaming каналов: {e}")
        return []


def get_cameras(ip):
    auth = httpx.DigestAuth(USER, PASSWORD)
    url = f"http://{ip}/ISAPI/Streaming/channels"
    try:
        with httpx.Client(auth=auth, verify=False, timeout=10) as client:
            response = client.get(url)
            if response.status_code == 200:
                return parse_streaming_channels(response.content)
            else:
                print(f"❌ HTTP {response.status_code} при получении камер с {ip}")
                return []
    except Exception as e:
        print(f"⚠️ Ошибка подключения к {ip}: {e}")
        return []


def parse_proxy_channels(xml_content):
    """Парсит XML с информацией о камерах и возвращает список с именами, IP и channel_id"""
    try:
        root = etree.fromstring(xml_content)
        for elem in root.iter():
            if elem.tag.startswith('{'):
                elem.tag = elem.tag.split('}', 1)[1]

        cameras = []
        for channel in root.xpath('.//InputProxyChannel'):
            ch_id_elem = channel.find('id')
            name_elem = channel.find('name')
            ip_elem = channel.find('.//ipAddress')

            if ch_id_elem is not None and ip_elem is not None:
                try:
                    ch_id = int(ch_id_elem.text)
                    name = name_elem.text.strip() if name_elem is not None and name_elem.text else f"Камера {ch_id}"
                    ip = ip_elem.text.strip()
                    if ip and ip != "0.0.0.0":
                        cameras.append({
                            'channel_id': ch_id,
                            'name': name,
                            'ip': ip
                        })
                except (ValueError, AttributeError):
                    continue
        cameras.sort(key=lambda x: x['channel_id'])
        return cameras
    except Exception as e:
        print(f"❌ Ошибка парсинга proxy channels: {e}")
        return []


def get_camera_list_from_nvr(ip):
    """Получает список камер с именем и IP адресом с одного NVR"""
    auth = httpx.DigestAuth(USER, PASSWORD)
    url = f"http://{ip}/ISAPI/ContentMgmt/InputProxy/channels"
    try:
        with httpx.Client(auth=auth, verify=False, timeout=10) as client:
            response = client.get(url)
            if response.status_code == 200:
                cameras = parse_proxy_channels(response.content)
                # Добавляем информацию о NVR к каждой камере
                for camera in cameras:
                    camera['nvr_ip'] = ip
                return cameras
            else:
                print(f"❌ HTTP {response.status_code} при запросе proxy channels с {ip}")
                return []
    except Exception as e:
        print(f"⚠️ Ошибка подключения к {ip}: {e}")
        return []


def get_all_cameras_from_all_nvrs():
    """Собирает список всех камер со всех NVR"""
    all_cameras = []
    for nvr_ip in NVR_IPS:
        cameras = get_camera_list_from_nvr(nvr_ip)
        all_cameras.extend(cameras)
    return all_cameras


def parse_search_results(xml_content):
    try:
        root = etree.fromstring(xml_content)
        for elem in root.iter():
            if elem.tag.startswith('{'):
                elem.tag = elem.tag.split('}', 1)[1]
        results = []
        for item in root.xpath('.//searchMatchItem'):
            time_span = item.find('timeSpan')
            start = time_span.findtext('startTime', default='N/A') if time_span is not None else 'N/A'
            end = time_span.findtext('endTime', default='N/A') if time_span is not None else 'N/A'
            media = item.find('mediaSegmentDescriptor')
            playback_uri = media.findtext('playbackURI', default='N/A') if media is not None else 'N/A'
            rec_type = "video"
            if media is not None:
                content_type = media.findtext('contentType', default='')
                if content_type:
                    rec_type = content_type
            duration_str = "00:00:00"
            if start != 'N/A' and end != 'N/A':
                try:
                    s = datetime.fromisoformat(start.replace('Z', ''))
                    e = datetime.fromisoformat(end.replace('Z', ''))
                    delta = e - s
                    total_sec = int(delta.total_seconds())
                    h, remainder = divmod(total_sec, 3600)
                    m, s = divmod(remainder, 60)
                    duration_str = f"{h:02}:{m:02}:{s:02}"
                except:
                    pass
            results.append({
                'start': start,
                'end': end,
                'type': rec_type,
                'playbackURI': playback_uri,
                'duration': duration_str
            })
        return results
    except Exception as e:
        print(f"❌ Ошибка парсинга записей: {e}")
        return []


def search_recordings_page(ip, track_id, start_time, end_time, max_results=100):
    auth = httpx.DigestAuth(USER, PASSWORD)
    url = f"http://{ip}/ISAPI/ContentMgmt/search"
    search_id = str(uuid.uuid4()).upper()
    xml_body = (
        '<CMSearchDescription xmlns="http://www.hikvision.com/ver20/XMLSchema">'
        f'<searchID>{search_id}</searchID>'
        '<trackList>'
        f'<trackID>{track_id}</trackID>'
        '</trackList>'
        '<timeSpanList>'
        '<timeSpan>'
        f'<startTime>{start_time}</startTime>'
        f'<endTime>{end_time}</endTime>'
        '</timeSpan>'
        '</timeSpanList>'
        f'<maxResults>{max_results}</maxResults>'
        '<searchResultPosition>0</searchResultPosition>'
        '</CMSearchDescription>'
    )

    try:
        with httpx.Client(auth=auth, verify=False, timeout=15) as client:
            headers = {"Content-Type": "application/xml"}
            response = client.post(url, content=xml_body.encode("utf-8"), headers=headers)
            if response.status_code == 200:
                return parse_search_results(response.content)
            else:
                print(f"❌ Ошибка поиска: {response.status_code}")
                return None
    except Exception as e:
        print(f"⚠️ Исключение при поиске: {e}")
        return None


def search_recordings_all(ip, track_id, start_time, end_time, max_records=64):
    all_results = []
    _recursive_search(ip, track_id, start_time, end_time, all_results, max_records)
    
    seen = set()
    unique_results = []
    for r in all_results:
        key = (r['start'], r['end'], r['playbackURI'])
        if key not in seen:
            seen.add(key)
            unique_results.append(r)
    
    unique_results.sort(key=lambda x: x['start'] if x['start'] != 'N/A' else '')
    return unique_results


def _recursive_search(ip, track_id, start_time, end_time, results, max_records):
    recordings = search_recordings_page(ip, track_id, start_time, end_time, max_records)
    if recordings is None:
        return

    results.extend(recordings)

    if len(recordings) == max_records:
        try:
            start_dt = datetime.fromisoformat(start_time.replace('Z', ''))
            end_dt = datetime.fromisoformat(end_time.replace('Z', ''))
            if (end_dt - start_dt).total_seconds() <= 60:
                return
            mid_dt = start_dt + (end_dt - start_dt) / 2
            mid_time = mid_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

            _recursive_search(ip, track_id, start_time, mid_time, results, max_records)
            _recursive_search(ip, track_id, mid_time, end_time, results, max_records)
        except Exception as e:
            print(f"⚠️ Ошибка при разбиении интервала: {e}")


# ==============================
# Глобальные очереди и флаги
# ==============================
log_queue = queue.Queue()
preview_queue = queue.Queue(maxsize=1)
file_progress_queue = queue.Queue()
frame_progress_queue = queue.Queue()
speed_queue = queue.Queue()
final_log_path_queue = queue.Queue()

preview_enabled = False
preview_lock = threading.Lock()


def set_preview_enabled(enabled):
    global preview_enabled
    with preview_lock:
        preview_enabled = enabled


def is_preview_enabled():
    with preview_lock:
        return preview_enabled


def log_message(msg):
    if msg.strip():
        if not msg.endswith('\n'):
            msg += '\n'
        log_queue.put(msg)


# ==============================
# Поток для чтения кадров (CPU)
# ==============================
def frame_producer(vr, frame_indices, batch_size, out_queue, stop_event):
    try:
        total = len(frame_indices)
        for start in range(0, total, batch_size):
            if stop_event.is_set():
                break
            batch_indices = frame_indices[start:start + batch_size]
            try:
                frames = vr.get_batch(batch_indices).asnumpy()
                out_queue.put(('frames', batch_indices, frames))
            except Exception as e:
                out_queue.put(('error', f"Ошибка чтения кадров {batch_indices}: {e}"))
        out_queue.put(('done', None, None))
    except Exception as e:
        out_queue.put(('error', f"Критическая ошибка в producer: {e}"))


# ==============================
# Обработка одного видео с pipeline
# ==============================
def process_single_video(video_path, start_minute, skip_frames, batch_size, roi, min_presence_sec, common_log_path):
    video_name = os.path.basename(video_path)
    try:
        ROI = roi
        vr = VideoReader(video_path, ctx=cpu(0))
        orig_fps = 20
        if orig_fps <= 0:
            orig_fps = 20.0
        total_frames = len(vr)
        video_duration_sec = total_frames / orig_fps

        start_frame = int(start_minute * 60 * orig_fps)
        log_message(f"Обработка: {video_name}")
        log_message(f"FPS: {orig_fps:.2f}, Всего кадров: {total_frames}, Старт с: {start_frame}")
        log_message(f"Зона: x1={ROI[0]}, y1={ROI[1]}, x2={ROI[2]}, y2={ROI[3]}")

        model = YOLO("yolov8m.pt")

        def format_video_time(seconds):
            m = int(seconds // 60)
            s = int(seconds % 60)
            return f"{m}:{s:02d}"

        def log_event(msg):
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(common_log_path, 'a', encoding='utf-8') as f:
                f.write(f"[{timestamp}] {msg}\n")

        person_in_roi = False
        entry_frame = None
        processed_video_time = 0.0
        prev_time = time.time()
        start_real_time = time.time()
        log_event(f"Запуск обработки файла: {video_name}")

        frame_indices = list(range(start_frame, total_frames, skip_frames))
        total_frame_count = len(frame_indices)

        # Очередь для передачи батчей от producer к consumer
        frame_queue = queue.Queue(maxsize=3)  # буфер на 3 батча
        stop_event = threading.Event()

        # Запуск producer
        producer_thread = threading.Thread(
            target=frame_producer,
            args=(vr, frame_indices, batch_size, frame_queue, stop_event),
            daemon=True
        )
        producer_thread.start()

        processed_batches = 0
        total_batches = (total_frame_count + batch_size - 1) // batch_size

        while True:
            item = frame_queue.get()
            if item[0] == 'done':
                break
            elif item[0] == 'error':
                log_message(f"[WARN] {video_name}: {item[1]}")
                continue

            current_indices, raw_frames = item[1], item[2]
            processed_batches += 1
            current_frame_progress = min(processed_batches * batch_size, total_frame_count)
            frame_progress_queue.put((current_frame_progress, total_frame_count))

            # Фильтрация валидных кадров
            valid_frames = []
            valid_indices = []
            for i, frame in enumerate(raw_frames):
                if frame is None or frame.size == 0 or frame.shape[0] == 0 or frame.shape[1] == 0:
                    log_message(f"[WARN] Пустой кадр в {video_name} на индексе {current_indices[i]}")
                    continue
                valid_frames.append(frame)
                valid_indices.append(current_indices[i])

            if not valid_frames:
                continue

            # Inference на GPU (или CPU)
            try:
                results = model(valid_frames, classes=[0], verbose=False)
            except Exception as e:
                log_message(f"[ERROR] YOLO ошибка на батче {valid_indices}: {e}")
                continue

            # Расчёт скорости
            curr_time = time.time()
            real_elapsed = curr_time - prev_time
            prev_time = curr_time

            if real_elapsed > 0:
                video_time = len(valid_indices) * skip_frames / orig_fps
                processed_video_time += video_time
                speedup = video_time / real_elapsed
                speed_queue.put(speedup)
                log_message(f"[PROGRESS] {video_name} | Speed: {speedup:.1f}x | Обработано: {format_video_time(processed_video_time)}")

            # Обработка результатов и превью
            for frame_idx, result in zip(valid_indices, results):
                frame_rgb = result.orig_img
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                boxes = result.boxes.xyxy.cpu().numpy()
                current_in_roi = False

                cv2.rectangle(frame_bgr, (ROI[0], ROI[1]), (ROI[2], ROI[3]), (0, 0, 255), 2)

                for box in boxes:
                    x1, y1, x2, y2 = map(int, box)
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    in_roi = ROI[0] <= cx <= ROI[2] and ROI[1] <= cy <= ROI[3]
                    if in_roi:
                        current_in_roi = True
                    color = (0, 255, 255) if in_roi else (255, 255, 0)
                    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
                    if in_roi:
                        cv2.putText(frame_bgr, "Detected!", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                if is_preview_enabled():
                    try:
                        if not preview_queue.empty():
                            preview_queue.get_nowait()
                        preview_queue.put_nowait(frame_bgr.copy())
                    except:
                        pass

                if not person_in_roi and current_in_roi:
                    person_in_roi = True
                    entry_frame = frame_idx
                elif person_in_roi and not current_in_roi:
                    person_in_roi = False
                    start_sec = entry_frame / orig_fps
                    end_sec = frame_idx / orig_fps
                    duration = end_sec - start_sec
                    if duration >= min_presence_sec:
                        log_event(f"Обнаружен человек в зоне. {format_video_time(start_sec)} <-> {format_video_time(end_sec)}")
                        log_message(f"[DETECT] {video_name}: {format_video_time(start_sec)} <-> {format_video_time(end_sec)}")
                    entry_frame = None

        if person_in_roi and entry_frame is not None:
            start_sec = entry_frame / orig_fps
            end_sec = (total_frames - 1) / orig_fps
            duration = end_sec - start_sec
            if duration >= min_presence_sec:
                log_event(f"Обнаружен человек в зоне. {format_video_time(start_sec)} <-> {format_video_time(end_sec)}")
                log_message(f"[DETECT] {video_name}: {format_video_time(start_sec)} <-> {format_video_time(end_sec)}")

        total_real = time.time() - start_real_time
        minutes = int(total_real // 60)
        seconds = int(total_real % 60)
        video_duration_str = format_video_time(video_duration_sec)
        log_message(f"Завершено: {video_name} | Время: {minutes}:{seconds:02d} | Длительность видео: {video_duration_str}")

        return video_duration_sec, total_real

    except Exception as e:
        error_msg = f"[ERROR] {video_name}: {str(e)}"
        log_message(error_msg)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(common_log_path, 'a', encoding='utf-8') as f:
            f.write(f"[{timestamp}] {error_msg}\n")
        return None
    finally:
        set_preview_enabled(False)
        speed_queue.put(0.0)
        frame_progress_queue.put((0, 0))


# ==============================
# Обработка всех файлов в папке
# ==============================
def process_folder(folder_path, start_minute, skip_frames, batch_size, roi, min_presence_sec):
    mp4_files = sorted(glob.glob(os.path.join(folder_path, "*.mp4")))
    if not mp4_files:
        log_message("[WARN] В папке не найдено ни одного .mp4 файла!")
        return

    parent_dir = os.path.dirname(folder_path)
    folder_name = os.path.basename(folder_path)
    common_log_path = os.path.join(parent_dir, f"{folder_name}.txt")

    with open(common_log_path, 'w', encoding='utf-8') as f:
        f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Начало обработки\n")

    log_message(f"Найдено {len(mp4_files)} файлов. Лог: {os.path.basename(common_log_path)}")
    file_progress_queue.put((0, len(mp4_files)))

    total_video_duration = 0.0
    total_processing_time = 0.0
    processed_count = 0
    start_overall = time.time()

    for idx, video_path in enumerate(mp4_files, start=1):
        set_preview_enabled(True)
        result = process_single_video(video_path, start_minute, skip_frames, batch_size, roi, min_presence_sec, common_log_path)
        if result is not None:
            video_dur, proc_time = result
            total_video_duration += video_dur
            total_processing_time += proc_time
            processed_count += 1
            file_progress_queue.put((idx, len(mp4_files)))
        time.sleep(0.5)

    file_progress_queue.put((len(mp4_files), len(mp4_files)))
    frame_progress_queue.put((0, 0))
    speed_queue.put(0.0)
    set_preview_enabled(False)
    final_log_path_queue.put(common_log_path)

    def format_hhmmss(seconds):
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        return f"{h}:{m:02d}:{s:02d}" if h > 0 else f"{m}:{s:02d}"

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    summary_lines = [
        f"[{timestamp}] [ИТОГО] Всего обработано файлов: {processed_count}",
        f"[{timestamp}] [ИТОГО] Суммарная длительность видео: {format_hhmmss(total_video_duration)} ({int(total_video_duration)} сек)",
        f"[{timestamp}] [ИТОГО] Общее время обработки: {format_hhmmss(time.time() - start_overall)} ({int(time.time() - start_overall)} сек)",
        f"[{timestamp}] [ИТОГО] Средняя скорость обработки: {total_video_duration / (time.time() - start_overall):.1f}x"
    ]

    with open(common_log_path, 'a', encoding='utf-8') as f:
        for line in summary_lines:
            f.write(line + "\n")
    for line in summary_lines:
        log_message(line.replace(f"[{timestamp}] ", ""))


# ==============================
# GUI
# ==============================
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Анализ видео")
        self.root.geometry("950x700")  # немного расширили окно
        self.root.resizable(False, False)

        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() // 2) - (950 // 2)
        y = (self.root.winfo_screenheight() // 2) - (700 // 2)
        self.root.geometry(f"950x700+{x}+{y}")

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Получаем путь к рабочему столу текущего пользователя
        desktop_path = os.path.join(os.path.expanduser('~'), 'Desktop')
        self.folder_path = tk.StringVar(value=desktop_path)
        self.start_minute = tk.StringVar(value="0")
        self.skip_frames = tk.StringVar(value="10")
        self.batch_size = tk.StringVar(value="16")
        self.min_presence_sec = tk.StringVar(value="0")  # НОВОЕ ПОЛЕ
        self.roi = (1144, 724, 1836, 1386)

        # Выбор папки
        folder_frame = tk.Frame(root)
        folder_frame.pack(pady=5, padx=10, fill=tk.X)
        tk.Label(folder_frame, text="Папка:").pack(side=tk.LEFT)
        tk.Entry(folder_frame, textvariable=self.folder_path, width=50).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        tk.Button(folder_frame, text="Загрузить с регистратора", command=self.open_hikvision_downloader).pack(side=tk.RIGHT, padx=(5, 0))
        tk.Button(folder_frame, text="Обзор...", command=self.browse_folder).pack(side=tk.RIGHT)

        # Настройки
        settings_frame = tk.Frame(root)
        settings_frame.pack(pady=5, padx=10, fill=tk.X)
        tk.Label(settings_frame, text="Старт (мин):").grid(row=0, column=0, sticky=tk.W)
        tk.Entry(settings_frame, textvariable=self.start_minute, width=10).grid(row=0, column=1, padx=5)
        tk.Label(settings_frame, text="Пропуск кадров:").grid(row=0, column=2, sticky=tk.W)
        tk.Entry(settings_frame, textvariable=self.skip_frames, width=10).grid(row=0, column=3, padx=5)
        tk.Label(settings_frame, text="Размер батча:").grid(row=0, column=4, sticky=tk.W)
        tk.Entry(settings_frame, textvariable=self.batch_size, width=10).grid(row=0, column=5, padx=5)
        tk.Label(settings_frame, text="Присутствие больше (сек):").grid(row=0, column=6, sticky=tk.W, padx=(10, 0))
        tk.Entry(settings_frame, textvariable=self.min_presence_sec, width=10).grid(row=0, column=7, padx=5)

        # Кнопки
        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=10)
        tk.Button(btn_frame, text="Выбрать зону", command=self.select_roi,
                  bg="lightgray", fg="black", width=16, height=2, font=("Arial", 10)).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="СТАРТ", command=self.start_processing,
                  bg="lightgray", fg="black", width=16, height=2, font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=5)

        # Прогресс по файлам
        file_progress_frame = tk.Frame(root)
        file_progress_frame.pack(fill=tk.X, padx=10, pady=2)
        self.file_label = tk.Label(file_progress_frame, text="Обработано 0 из 0 файлов", anchor="w")
        self.file_label.pack(side=tk.LEFT)
        self.file_canvas = tk.Canvas(file_progress_frame, width=200, height=20, bg='white', highlightthickness=0)
        self.file_canvas.pack(side=tk.RIGHT)
        self.file_progress_rect = self.file_canvas.create_rectangle(0, 0, 0, 20, fill='green')
        self.file_percent_text = self.file_canvas.create_text(100, 10, text="0%", fill="black", font=("Arial", 8))

        # Прогресс по кадрам
        frame_progress_frame = tk.Frame(root)
        frame_progress_frame.pack(fill=tk.X, padx=10, pady=2)
        self.frame_label = tk.Label(frame_progress_frame, text="Обработано 0 из 0 кадров", anchor="w")
        self.frame_label.pack(side=tk.LEFT)
        self.frame_canvas = tk.Canvas(frame_progress_frame, width=200, height=20, bg='white', highlightthickness=0)
        self.frame_canvas.pack(side=tk.RIGHT)
        self.frame_progress_rect = self.frame_canvas.create_rectangle(0, 0, 0, 20, fill='blue')
        self.frame_percent_text = self.frame_canvas.create_text(100, 10, text="0%", fill="black", font=("Arial", 8))

        # Скорость
        speed_frame = tk.Frame(root)
        speed_frame.pack(fill=tk.X, padx=10, pady=2)
        self.speed_label = tk.Label(speed_frame, text="Скорость обработки 0.0x", anchor="w")
        self.speed_label.pack(side=tk.LEFT)

        # Превью
        preview_frame = tk.Frame(root, relief=tk.SUNKEN, bd=1)
        preview_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.preview_canvas = tk.Canvas(preview_frame, bg='black')
        self.preview_canvas.pack(fill=tk.BOTH, expand=True)

        self.process_log_queue()
        self.process_file_progress_queue()
        self.process_frame_progress_queue()
        self.process_speed_queue()
        self.process_final_log_open()
        self.update_preview()

    def process_log_queue(self):
        try:
            while True:
                _ = log_queue.get_nowait()
        except queue.Empty:
            pass
        self.root.after(100, self.process_log_queue)

    def process_file_progress_queue(self):
        try:
            while True:
                current, total = file_progress_queue.get_nowait()
                if total > 0:
                    percent = int((current / total) * 100)
                    self.file_label.config(text=f"Обработано {current} из {total} файлов")
                    self.file_canvas.coords(self.file_progress_rect, 0, 0, percent * 2, 20)
                    self.file_canvas.itemconfig(self.file_percent_text, text=f"{percent}%")
                else:
                    self.file_label.config(text="Обработано 0 из 0 файлов")
                    self.file_canvas.coords(self.file_progress_rect, 0, 0, 0, 20)
                    self.file_canvas.itemconfig(self.file_percent_text, text="0%")
        except queue.Empty:
            pass
        self.root.after(100, self.process_file_progress_queue)

    def process_frame_progress_queue(self):
        try:
            while True:
                current, total = frame_progress_queue.get_nowait()
                if total > 0:
                    percent = int((current / total) * 100)
                    self.frame_label.config(text=f"Обработано {current} из {total} кадров")
                    self.frame_canvas.coords(self.frame_progress_rect, 0, 0, percent * 2, 20)
                    self.frame_canvas.itemconfig(self.frame_percent_text, text=f"{percent}%")
                else:
                    self.frame_label.config(text="Обработано 0 из 0 кадров")
                    self.frame_canvas.coords(self.frame_progress_rect, 0, 0, 0, 20)
                    self.frame_canvas.itemconfig(self.frame_percent_text, text="0%")
        except queue.Empty:
            pass
        self.root.after(100, self.process_frame_progress_queue)

    def process_speed_queue(self):
        try:
            while True:
                speed = speed_queue.get_nowait()
                self.speed_label.config(text=f"Скорость обработки {speed:.1f}x")
        except queue.Empty:
            pass
        self.root.after(100, self.process_speed_queue)

    def process_final_log_open(self):
        try:
            while True:
                log_path = final_log_path_queue.get_nowait()
                if os.path.exists(log_path):
                    system = platform.system()
                    if system == "Windows":
                        os.startfile(log_path)
                    elif system == "Darwin":
                        subprocess.Popen(["open", log_path])
                    else:
                        subprocess.Popen(["xdg-open", log_path])
        except queue.Empty:
            pass
        self.root.after(500, self.process_final_log_open)

    def browse_folder(self):
        current_path = self.folder_path.get().strip()
        initialdir = current_path if os.path.isdir(current_path) else None
        folder = filedialog.askdirectory(title="Выберите папку с видеофайлами", initialdir=initialdir)
        if folder:
            self.folder_path.set(folder)

    def select_roi(self):
        folder = self.folder_path.get().strip()
        if not folder or not os.path.isdir(folder):
            messagebox.showwarning("Ошибка", "Сначала выберите папку с видео!")
            return

        mp4_files = glob.glob(os.path.join(folder, "*.mp4"))
        if not mp4_files:
            messagebox.showwarning("Ошибка", "В папке нет .mp4 файлов!")
            return

        first_video = mp4_files[0]
        try:
            vr = VideoReader(first_video, ctx=cpu(0))
            frame = vr[0].asnumpy()
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось прочитать первый кадр: {e}")
            return

        orig_h, orig_w = frame.shape[:2]
        display_w, display_h = 1280, 720
        pil_image = Image.fromarray(frame)
        scale = min(display_w / orig_w, display_h / orig_h)
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)
        resized_img = pil_image.resize((new_w, new_h), Image.LANCZOS)

        roi_window = tk.Toplevel(self.root)
        roi_window.title("Выберите зону (перетащите мышью)")
        roi_window.geometry(f"{display_w}x{display_h + 40}")
        roi_window.resizable(False, False)
        roi_window.update_idletasks()
        x = (roi_window.winfo_screenwidth() // 2) - (display_w // 2)
        y = (roi_window.winfo_screenheight() // 2) - (display_h // 2) - 20
        roi_window.geometry(f"{display_w}x{display_h + 40}+{x}+{y}")

        canvas = tk.Canvas(roi_window, width=display_w, height=display_h, bg='black')
        canvas.pack()
        offset_x = (display_w - new_w) // 2
        offset_y = (display_h - new_h) // 2
        photo = ImageTk.PhotoImage(resized_img)
        canvas.create_image(offset_x, offset_y, anchor=tk.NW, image=photo)
        canvas.image = photo

        rect_id = None
        start_x = start_y = None

        def on_mouse_down(event):
            nonlocal start_x, start_y, rect_id
            start_x, start_y = event.x, event.y
            if rect_id:
                canvas.delete(rect_id)
            rect_id = canvas.create_rectangle(start_x, start_y, start_x, start_y, outline='red', width=2)

        def on_mouse_move(event):
            nonlocal rect_id
            if rect_id:
                canvas.coords(rect_id, start_x, start_y, event.x, event.y)

        def on_mouse_up(event):
            nonlocal start_x, start_y
            x1_disp, y1_disp = min(start_x, event.x), min(start_y, event.y)
            x2_disp, y2_disp = max(start_x, event.x), max(start_y, event.y)
            if x2_disp - x1_disp > 5 and y2_disp - y1_disp > 5:
                x1_orig = max(0, int((x1_disp - offset_x) / scale))
                y1_orig = max(0, int((y1_disp - offset_y) / scale))
                x2_orig = min(orig_w, int((x2_disp - offset_x) / scale))
                y2_orig = min(orig_h, int((y2_disp - offset_y) / scale))
                if x2_orig > x1_orig and y2_orig > y1_orig:
                    self.roi = (x1_orig, y1_orig, x2_orig, y2_orig)
                    log_message(f"[ROI] Выбрана зона: {self.roi}")
                else:
                    messagebox.showwarning("Внимание", "Некорректная зона. Попробуйте снова.")
            roi_window.destroy()

        canvas.bind("<Button-1>", on_mouse_down)
        canvas.bind("<B1-Motion>", on_mouse_move)
        canvas.bind("<ButtonRelease-1>", on_mouse_up)

    def start_processing(self):
        try:
            folder = self.folder_path.get().strip()
            if not folder or not os.path.isdir(folder):
                messagebox.showwarning("Ошибка", "Укажите корректную папку с видеофайлами!")
                return
            start_min = int(self.start_minute.get())
            skip = int(self.skip_frames.get())
            batch = int(self.batch_size.get())
            min_sec = float(self.min_presence_sec.get())  # может быть дробным
            if skip < 1 or batch < 1 or min_sec < 0:
                messagebox.showwarning("Ошибка", "Некорректные параметры!")
                return
            thread = threading.Thread(
                target=process_folder,
                args=(folder, start_min, skip, batch, self.roi, min_sec),
                daemon=True
            )
            thread.start()
        except ValueError:
            messagebox.showerror("Ошибка", "Проверьте, что все параметры — числа.")

    def update_preview(self):
        try:
            if not preview_queue.empty():
                frame = preview_queue.get_nowait()
                canvas_width = self.preview_canvas.winfo_width()
                canvas_height = self.preview_canvas.winfo_height()
                if canvas_width <= 1 or canvas_height <= 1:
                    canvas_width, canvas_height = 800, 450
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w = frame_rgb.shape[:2]
                scale = min(canvas_width / w, canvas_height / h)
                new_w, new_h = int(w * scale), int(h * scale)
                resized = cv2.resize(frame_rgb, (new_w, new_h))
                img = Image.fromarray(resized)
                photo = ImageTk.PhotoImage(image=img)
                self.preview_canvas.delete("all")
                x = (canvas_width - new_w) // 2
                y = (canvas_height - new_h) // 2
                self.preview_canvas.create_image(x, y, anchor=tk.NW, image=photo)
                self.preview_canvas.image = photo
        except Exception:
            pass
        self.root.after(50, self.update_preview)

    def on_closing(self):
        set_preview_enabled(False)
        self.root.destroy()

    def open_hikvision_downloader(self):
        """Открывает окно загрузки видео с регистраторов Hikvision"""
        downloader_window = tk.Toplevel(self.root)
        downloader_window.transient(self.root)
        downloader_app = HikvisionDownloaderApp(downloader_window)


# ==============================
# Класс HikvisionDownloaderApp
# ==============================
class HikvisionDownloaderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Загрузить с регистратора")
        self.root.resizable(False, False)

        window_width = 720
        window_height = 520
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        x = (screen_width // 2) - (window_width // 2)
        y = (screen_height // 2) - (window_height // 2)
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")

        self.selected_camera_name = tk.StringVar(value="")
        self.recordings = []
        self.selected_recording = tk.StringVar()
        self.all_cameras_list = []  # Список всех камер со всех NVR: [{name, nvr_ip, channel_id, ip}, ...]
        self.camera_name_to_info = {}  # Словарь для быстрого поиска информации о камере по имени
        self.selected_camera_info = None  # Информация о выбранной камере

        # Название камеры
        ttk.Label(root, text="Название камеры:").grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.camera_name_combo = ttk.Combobox(root, textvariable=self.selected_camera_name, values=[], state="disabled", width=50)
        self.camera_name_combo.grid(row=0, column=1, columnspan=3, padx=5, pady=10, sticky="ew")
        self.camera_name_combo.bind("<<ComboboxSelected>>", self.on_camera_name_selected)

        ttk.Label(root, text="Начало:").grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self.start_date = ttk.Entry(root, width=12)
        self.start_date.grid(row=1, column=1, padx=5, pady=5, sticky="w")
        self.start_time = ttk.Entry(root, width=8)
        self.start_time.grid(row=1, column=2, padx=5, pady=5, sticky="w")

        ttk.Label(root, text="Конец:").grid(row=2, column=0, padx=10, pady=5, sticky="w")
        self.end_date = ttk.Entry(root, width=12)
        self.end_date.grid(row=2, column=1, padx=5, pady=5, sticky="w")
        self.end_time = ttk.Entry(root, width=8)
        self.end_time.grid(row=2, column=2, padx=5, pady=5, sticky="w")

        today = date.today().strftime("%Y-%m-%d")
        self.start_date.insert(0, today)
        self.start_time.insert(0, "00:00:00")
        self.end_date.insert(0, today)
        self.end_time.insert(0, "23:59:59")

        self.load_segments_btn = ttk.Button(root, text="Загрузить записи", command=self.load_segments, state="disabled")
        self.load_segments_btn.grid(row=1, column=3, rowspan=2, padx=10, pady=10, sticky="ns")

        ttk.Label(root, text="Доступные записи:").grid(row=3, column=0, padx=10, pady=10, sticky="w")
        self.recordings_combo = ttk.Combobox(root, textvariable=self.selected_recording, state="readonly", width=85)
        self.recordings_combo.grid(row=4, column=0, columnspan=4, padx=10, pady=5, sticky="ew")

        # Две кнопки в одной строке
        button_frame = ttk.Frame(root)
        button_frame.grid(row=5, column=0, columnspan=4, pady=15)
        self.download_btn = ttk.Button(button_frame, text="Загрузить видео", command=self.download_video)
        self.download_btn.pack(side="left", padx=(0, 10))
        self.download_all_btn = ttk.Button(button_frame, text="Загрузить ВСЕ видео", command=self.download_all_videos)
        self.download_all_btn.pack(side="left")

        self.progress_frame = tk.Frame(root)
        self.progress_frame.grid(row=6, column=0, columnspan=4, pady=5, sticky="ew")
        self.progress_frame.columnconfigure(0, weight=1)

        self.progress_canvas = tk.Canvas(self.progress_frame, height=25, bg="white", relief="sunken", bd=1)
        self.progress_canvas.grid(row=0, column=0, sticky="ew")

        # Инициализация элементов прогресс-бара с правильным z-order
        self.progress_canvas.create_rectangle(0, 0, 0, 25, fill="green", tags="progress_fill")
        self.progress_text = self.progress_canvas.create_text(
            0, 0, text="0%", fill="black", font=("TkDefaultFont", 10), tags="progress_text"
        )
        self._update_progress_bar(0)  # инициализация

        self.size_label = ttk.Label(root, text="Загрузка списка камер...")
        self.size_label.grid(row=7, column=0, columnspan=4, pady=5)

        root.columnconfigure(0, weight=1)
        root.columnconfigure(1, weight=1)
        
        # Автоматически запускаем загрузку камер при открытии окна
        self.root.after(100, self.request_all_cameras)

    def _update_progress_bar(self, percent):
        """Обновляет только визуальные элементы прогресс-бара без изменения логики."""
        self.root.update_idletasks()
        width = self.progress_canvas.winfo_width()
        if width <= 1:
            width = self.progress_canvas.winfo_reqwidth()

        fill_width = int(width * (percent / 100))
        self.progress_canvas.coords("progress_fill", 0, 0, fill_width, 25)
        self.progress_canvas.coords("progress_text", width // 2, 12)
        self.progress_canvas.itemconfig("progress_text", text=f"{int(percent)}%")

    def request_all_cameras(self):
        """Запрашивает список всех камер со всех NVR"""
        self.camera_name_combo.config(state="disabled")
        self.size_label.config(text="Запрос списка камер...")
        threading.Thread(target=self._fetch_all_cameras, daemon=True).start()

    def _fetch_all_cameras(self):
        """Собирает все камеры со всех NVR в отдельном потоке"""
        all_cameras = get_all_cameras_from_all_nvrs()
        self.root.after(0, self._update_all_cameras_list, all_cameras)

    def _update_all_cameras_list(self, cameras):
        """Обновляет список камер в интерфейсе"""
        self.all_cameras_list = cameras
        
        if cameras:
            # Создаем список имен камер для отображения
            camera_names = []
            self.camera_name_to_info = {}
            
            for cam in cameras:
                # Формируем отображаемое имя: "Имя камеры (NVR_IP)"
                display_name = f"{cam['name']} ({cam['nvr_ip']})"
                camera_names.append(display_name)
                # Сохраняем информацию о камере для быстрого доступа
                self.camera_name_to_info[display_name] = cam
            
            self.camera_name_combo["values"] = camera_names
            self.camera_name_combo.config(state="readonly")
            self.size_label.config(text=f"Найдено {len(cameras)} камер")
        else:
            self.camera_name_combo["values"] = []
            self.camera_name_combo.config(state="readonly")
            self.size_label.config(text="Готово к загрузке")
            messagebox.showwarning("Внимание", "Не удалось получить список камер с регистраторов")

    def on_camera_name_selected(self, event=None):
        """Обрабатывает выбор камеры по имени - сохраняет информацию о камере"""
        selected_name = self.selected_camera_name.get()
        if not selected_name or selected_name not in self.camera_name_to_info:
            self.selected_camera_info = None
            self.load_segments_btn.config(state="disabled")
            return
        
        # Сохраняем информацию о выбранной камере
        self.selected_camera_info = self.camera_name_to_info[selected_name]
        # Активируем кнопку загрузки записей
        self.load_segments_btn.config(state="normal")

    def update_progress(self, percent, downloaded_mb=0, total_mb=0, mode="bytes"):
        if mode == "bytes":
            self.size_label.config(text=f"{downloaded_mb:.1f} из {total_mb:.1f} МБ")
        elif mode == "files":
            current = int((percent / 100) * self._total_files)
            self.size_label.config(text=f"Файлов: {current} из {self._total_files}")
        self._update_progress_bar(percent)

    def load_segments(self):
        if not self.selected_camera_info:
            messagebox.showerror("Ошибка", "Выберите камеру")
            return

        # Получаем информацию о выбранной камере
        nvr_ip = self.selected_camera_info['nvr_ip']
        channel_id = self.selected_camera_info['channel_id']
        
        # Вычисляем track_id на основе channel_id
        # Обычно track_id = channel_id * 100 + 1, но может отличаться в зависимости от системы
        # Попробуем сначала использовать channel_id напрямую, как номер камеры
        track_id = channel_id * 100 + 1

        start_iso = f"{self.start_date.get()}T{self.start_time.get()}Z"
        end_iso = f"{self.end_date.get()}T{self.end_time.get()}Z"

        try:
            datetime.fromisoformat(start_iso.replace('Z', ''))
            datetime.fromisoformat(end_iso.replace('Z', ''))
        except ValueError:
            messagebox.showerror("Ошибка", "Неверный формат даты или времени")
            return

        self.load_segments_btn.config(state="disabled")
        self.recordings_combo.set("")
        self.recordings = []
        self.size_label.config(text="Поиск записей...")

        threading.Thread(target=self._fetch_recordings, args=(nvr_ip, track_id, start_iso, end_iso), daemon=True).start()

    def _fetch_recordings(self, ip, track_id, start_iso, end_iso):
        recordings = search_recordings_all(ip, track_id, start_iso, end_iso)
        self.root.after(0, self._update_recordings_ui, recordings)

    def _update_recordings_ui(self, recordings):
        self.load_segments_btn.config(state="normal")
        if not recordings:
            messagebox.showinfo("Информация", "Записи не найдены")
            self.recordings_combo["values"] = []
            self.size_label.config(text="Готово к загрузке")
            return

        self.recordings = recordings
        display_list = []
        for rec in recordings:
            start_clean = rec['start'].replace('Z', '').replace('T', ' ') if rec['start'] != 'N/A' else '—'
            end_clean = rec['end'].replace('Z', '').replace('T', ' ') if rec['end'] != 'N/A' else '—'
            rec_type = rec['type'] if rec['type'] != 'N/A' else '—'
            duration = rec['duration']
            display = f"{start_clean} — {end_clean} | {rec_type} | {duration}"
            display_list.append(display)

        self.recordings_combo["values"] = display_list
        if display_list:
            self.recordings_combo.current(0)
        self.size_label.config(text=f"Найдено {len(recordings)} записей")

    def download_video(self):
        if not self.recordings:
            messagebox.showwarning("Предупреждение", "Сначала загрузите список записей")
            return

        if not self.selected_camera_info:
            messagebox.showerror("Ошибка", "Камера не выбрана")
            return

        idx = self.recordings_combo.current()
        if idx == -1:
            messagebox.showwarning("Предупреждение", "Выберите запись для загрузки")
            return

        rtsp_url = self.recordings[idx]['playbackURI']
        if not rtsp_url or rtsp_url == 'N/A':
            messagebox.showerror("Ошибка", "Недопустимый URI записи")
            return

        filepath = filedialog.asksaveasfilename(
            defaultextension=".mp4",
            filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")],
            title="Сохранить видео как..."
        )
        if not filepath:
            return

        nvr_ip = self.selected_camera_info['nvr_ip']
        self.download_btn.config(state="disabled")
        self.download_all_btn.config(state="disabled")
        self.update_progress(0, 0, 0, "bytes")

        threading.Thread(target=self._download_file, args=(nvr_ip, rtsp_url, filepath), daemon=True).start()

    def _download_file(self, ip, rtsp_url, filepath):
        auth = httpx.DigestAuth(USER, PASSWORD)
        url = f"http://{ip}/ISAPI/ContentMgmt/download?format=mp4"
        xml_body = (
            '<downloadRequest version="1.0" xmlns="http://www.isapi.org/ver20/XMLSchema">'
            f'<playbackURI>{rtsp_url}</playbackURI>'
            '</downloadRequest>'
        )

        try:
            with httpx.Client(auth=auth, verify=False, timeout=120) as client:
                headers = {"Content-Type": "application/xml"}
                with client.stream("POST", url, content=xml_body.encode("utf-8"), headers=headers) as response:
                    if response.status_code != 200:
                        self.root.after(0, self._download_error, f"HTTP {response.status_code}")
                        return

                    total_bytes = response.headers.get("content-length")
                    total_bytes = int(total_bytes) if total_bytes else None
                    total_mb = total_bytes / (1024 * 1024) if total_bytes else 0

                    downloaded = 0
                    with open(filepath, "wb") as f:
                        for chunk in response.iter_bytes(chunk_size=8192):
                            f.write(chunk)
                            downloaded += len(chunk)
                            downloaded_mb = downloaded / (1024 * 1024)
                            percent = (downloaded / total_bytes * 100) if total_bytes else 0
                            self.root.after(0, self.update_progress, percent, downloaded_mb, total_mb, "bytes")

            self.root.after(0, self._download_finished, True, filepath)
        except Exception as e:
            self.root.after(0, self._download_error, str(e))

    def _download_finished(self, success, filepath):
        self.download_btn.config(state="normal")
        self.download_all_btn.config(state="normal")
        if success:
            messagebox.showinfo("Успех", f"Видео сохранено:\n{filepath}")
        else:
            messagebox.showerror("Ошибка", "Не удалось скачать видео")

    def _download_error(self, error_msg):
        self.download_btn.config(state="normal")
        self.download_all_btn.config(state="normal")
        messagebox.showerror("Ошибка загрузки", f"Не удалось скачать видео:\n{error_msg}")

    def download_all_videos(self):
        if not self.recordings:
            messagebox.showwarning("Предупреждение", "Сначала загрузите список записей")
            return

        if not self.selected_camera_info:
            messagebox.showerror("Ошибка", "Камера не выбрана")
            return

        folderpath = filedialog.askdirectory(title="Выберите папку для сохранения всех видео")
        if not folderpath:
            return

        nvr_ip = self.selected_camera_info['nvr_ip']
        self.download_btn.config(state="disabled")
        self.download_all_btn.config(state="disabled")
        self._total_files = len(self.recordings)
        self.update_progress(0, 0, 0, "files")

        threading.Thread(target=self._download_all_files, args=(nvr_ip, folderpath), daemon=True).start()

    def _download_all_files(self, ip, folderpath):
        auth = httpx.DigestAuth(USER, PASSWORD)
        base_url = f"http://{ip}/ISAPI/ContentMgmt/download?format=mp4"

        total_count = len(self.recordings)
        success_count = 0

        for idx, rec in enumerate(self.recordings):
            rtsp_url = rec['playbackURI']
            if not rtsp_url or rtsp_url == 'N/A':
                print(f"⚠️ Пропущена запись {idx+1}: недопустимый URI")
                percent = ((idx + 1) / total_count) * 100
                self.root.after(0, self.update_progress, percent, 0, 0, "files")
                continue

            if rec['start'] != 'N/A':
                safe_start = rec['start'].replace('Z', '').replace(':', '-').replace('T', '_')
            else:
                safe_start = f"rec_{idx+1:03d}"
            filename = f"{safe_start}.mp4"
            filepath = os.path.join(folderpath, filename)

            xml_body = (
                '<downloadRequest version="1.0" xmlns="http://www.isapi.org/ver20/XMLSchema">'
                f'<playbackURI>{rtsp_url}</playbackURI>'
                '</downloadRequest>'
            )

            try:
                with httpx.Client(auth=auth, verify=False, timeout=120) as client:
                    headers = {"Content-Type": "application/xml"}
                    with client.stream("POST", base_url, content=xml_body.encode("utf-8"), headers=headers) as response:
                        if response.status_code == 200:
                            with open(filepath, "wb") as f:
                                for chunk in response.iter_bytes(chunk_size=8192):
                                    f.write(chunk)
                            success_count += 1
                        else:
                            print(f"❌ Ошибка загрузки записи {idx+1}: HTTP {response.status_code}")
            except Exception as e:
                print(f"⚠️ Ошибка при загрузке записи {idx+1}: {e}")

            # Обновляем прогресс ПОСЛЕ обработки файла
            percent = ((idx + 1) / total_count) * 100
            self.root.after(0, self.update_progress, percent, 0, 0, "files")

        if hasattr(self, '_total_files'):
            del self._total_files

        self.root.after(0, self._all_download_finished, success_count, total_count)

    def _all_download_finished(self, success_count, total_count):
        self.download_btn.config(state="normal")
        self.download_all_btn.config(state="normal")
        if success_count == total_count:
            messagebox.showinfo("Успех", f"Все {success_count} видео успешно сохранены!")
        else:
            messagebox.showwarning("Завершено", f"Загружено {success_count} из {total_count} видео.\nНекоторые записи могли быть пропущены.")


def run():
    root = tk.Tk()
    app = App(root)
    root.mainloop()

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()