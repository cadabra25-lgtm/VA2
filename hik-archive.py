import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from datetime import datetime, date
import httpx
from lxml import etree
import threading
import uuid
import os


USER = "admin"
PASSWORD = "qweR-123"
NVR_IPS = ["172.28.10.21", "172.28.10.22", "172.28.10.23", "172.28.10.24", "172.28.10.25"]


# === API Functions ===
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


# === GUI ===
class HikvisionDownloaderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Hikvision Video Downloader")
        self.root.resizable(False, False)

        window_width = 720
        window_height = 540
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        x = (screen_width // 2) - (window_width // 2)
        y = (screen_height // 2) - (window_height // 2)
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")

        self.selected_nvr = tk.StringVar(value=NVR_IPS[0])
        self.selected_camera = tk.StringVar(value="")
        self.camera_numbers = []
        self.recordings = []
        self.selected_recording = tk.StringVar()

        ttk.Label(root, text="Регистратор:").grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.nvr_combo = ttk.Combobox(root, textvariable=self.selected_nvr, values=NVR_IPS, state="readonly", width=15)
        self.nvr_combo.grid(row=0, column=1, padx=5, pady=10, sticky="w")
        self.nvr_combo.bind("<<ComboboxSelected>>", self.on_nvr_selected)

        ttk.Label(root, text="Камера:").grid(row=0, column=2, padx=10, pady=10, sticky="w")
        self.camera_combo = ttk.Combobox(root, textvariable=self.selected_camera, values=[], state="disabled", width=5)
        self.camera_combo.grid(row=0, column=3, padx=5, pady=10, sticky="w")

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

        self.size_label = ttk.Label(root, text="Готово к загрузке")
        self.size_label.grid(row=7, column=0, columnspan=4, pady=5)

        root.columnconfigure(0, weight=1)

        self.root.after(100, self.on_nvr_selected)

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

    def on_nvr_selected(self, event=None):
        ip = self.selected_nvr.get()
        self.load_segments_btn.config(state="disabled")
        self.camera_combo.config(state="disabled")
        self.camera_combo.set("")
        threading.Thread(target=self._fetch_cameras, args=(ip,), daemon=True).start()

    def _fetch_cameras(self, ip):
        camera_numbers = get_cameras(ip)
        self.root.after(0, self._update_camera_list, camera_numbers)

    def _update_camera_list(self, camera_numbers):
        if camera_numbers:
            self.camera_numbers = camera_numbers
            camera_strs = [str(n) for n in camera_numbers]
            self.camera_combo["values"] = camera_strs
            self.camera_combo.set(camera_strs[0])
            self.camera_combo.config(state="readonly")
            self.load_segments_btn.config(state="normal")
        else:
            self.camera_numbers = []
            self.camera_combo["values"] = []
            self.camera_combo.set("")
            self.camera_combo.config(state="disabled")
            self.load_segments_btn.config(state="disabled")
            messagebox.showwarning("Внимание", f"Не удалось загрузить камеры с {self.selected_nvr.get()}")

    def update_progress(self, percent, downloaded_mb=0, total_mb=0, mode="bytes"):
        if mode == "bytes":
            self.size_label.config(text=f"{downloaded_mb:.1f} из {total_mb:.1f} МБ")
        elif mode == "files":
            current = int((percent / 100) * self._total_files)
            self.size_label.config(text=f"Файлов: {current} из {self._total_files}")
        self._update_progress_bar(percent)

    def load_segments(self):
        ip = self.selected_nvr.get()
        cam_str = self.selected_camera.get()
        if not cam_str or not self.camera_numbers:
            messagebox.showerror("Ошибка", "Нет доступных камер")
            return

        try:
            cam_num = int(cam_str)
            if cam_num not in self.camera_numbers:
                raise ValueError
            track_id = cam_num * 100 + 1
        except (ValueError, TypeError):
            messagebox.showerror("Ошибка", "Некорректный выбор камеры")
            return

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

        threading.Thread(target=self._fetch_recordings, args=(ip, track_id, start_iso, end_iso), daemon=True).start()

    def _fetch_recordings(self, ip, track_id, start_iso, end_iso):
        recordings = search_recordings_all(ip, track_id, start_iso, end_iso)
        self.root.after(0, self._update_recordings_ui, recordings)

    def _update_recordings_ui(self, recordings):
        self.load_segments_btn.config(state="normal")
        if not recordings:
            messagebox.showinfo("Информация", "Записи не найдены")
            self.recordings_combo["values"] = []
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

    def download_video(self):
        if not self.recordings:
            messagebox.showwarning("Предупреждение", "Сначала загрузите список записей")
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

        ip = self.selected_nvr.get()
        self.download_btn.config(state="disabled")
        self.download_all_btn.config(state="disabled")
        self.update_progress(0, 0, 0, "bytes")  # <<< ИСПРАВЛЕНО

        threading.Thread(target=self._download_file, args=(ip, rtsp_url, filepath), daemon=True).start()

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
                            self.root.after(0, self.update_progress, percent, downloaded_mb, total_mb, "bytes")  # <<< OK

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

        folderpath = filedialog.askdirectory(title="Выберите папку для сохранения всех видео")
        if not folderpath:
            return

        ip = self.selected_nvr.get()
        self.download_btn.config(state="disabled")
        self.download_all_btn.config(state="disabled")
        self._total_files = len(self.recordings)
        self.update_progress(0, 0, 0, "files")  # <<< ИСПРАВЛЕНО

        threading.Thread(target=self._download_all_files, args=(ip, folderpath), daemon=True).start()

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
                self.root.after(0, self.update_progress, percent, 0, 0, "files")  # <<< ИСПРАВЛЕНО
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
            self.root.after(0, self.update_progress, percent, 0, 0, "files")  # <<< ИСПРАВЛЕНО

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


if __name__ == "__main__":
    root = tk.Tk()
    app = HikvisionDownloaderApp(root)
    root.mainloop()