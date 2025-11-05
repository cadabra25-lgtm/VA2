import tkinter as tk
from tkinter import ttk, messagebox
import vlc
import httpx
from lxml import etree
import threading
import time


#eqweqweqwe


# === Настройки ===
USER = "admin"
PASSWORD = "qweR-123"
NVR_IPS = ["172.28.10.21", "172.28.10.22", "172.28.10.23", "172.28.10.24", "172.28.10.25"]
GRID_SIZES = [1, 4, 9, 16, 32]


# === Работа с NVR ===
def parse_proxy_channels(xml_content):
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
    auth = httpx.DigestAuth(USER, PASSWORD)
    url = f"http://{ip}/ISAPI/ContentMgmt/InputProxy/channels"
    try:
        with httpx.Client(auth=auth, verify=False, timeout=10) as client:
            response = client.get(url)
            if response.status_code == 200:
                return parse_proxy_channels(response.content)
            else:
                print(f"❌ HTTP {response.status_code} при запросе proxy channels с {ip}")
                return []
    except Exception as e:
        print(f"⚠️ Ошибка подключения к {ip}: {e}")
        return []


def get_rtsp_url_direct(camera_ip, stream_type="secondary"):
    channel = "102" if stream_type == "secondary" else "101"
    return f"rtsp://{USER}:{PASSWORD}@{camera_ip}:554/Streaming/channels/{channel}"


# === VLC Player ===
class VlcVideoPlayer:
    def __init__(self, rtsp_url, frame):
        self.rtsp_url = rtsp_url
        self.frame = frame
        self.instance = vlc.Instance(
            "--no-xlib",
            "--quiet",
            "--no-video-title-show",
            "--no-snapshot-preview",
            "--network-caching=300",
            "--rtsp-tcp"  # ← принудительно используем TCP (надёжнее)
        )
        self.player = self.instance.media_player_new()
        self.player.set_hwnd(frame.winfo_id())
        media = self.instance.media_new(rtsp_url)
        self.player.set_media(media)
        self.player.play()

    def release_later(self):
        """Вызывается в фоне через несколько секунд"""
        try:
            if self.player:
                self.player.stop()
                self.player.release()
            if self.instance:
                self.instance.release()
        except Exception as e:
            print(f"Ошибка при отложенной очистке: {e}")


# === Основное приложение ===
class MultiCameraViewerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Hikvision Multi-Camera Viewer (VLC)")
        self.root.state('zoomed')
        self.root.resizable(True, False)

        self.selected_nvr = tk.StringVar(value=NVR_IPS[0])
        self.selected_camera = tk.StringVar(value="")
        self.selected_grid = tk.StringVar(value="1")
        self.camera_list = []
        self.vlc_players = []
        self._redraw_scheduled = False

        # Верхняя панель
        control_frame = tk.Frame(root, bg="#2e2e2e", height=50)
        control_frame.pack(side="top", fill="x")
        control_frame.pack_propagate(False)

        ttk.Label(control_frame, text="Регистратор:", background="#2e2e2e", foreground="white").pack(side="left", padx=10)
        self.nvr_combo = ttk.Combobox(control_frame, textvariable=self.selected_nvr, values=NVR_IPS, state="readonly", width=15)
        self.nvr_combo.pack(side="left", padx=5)
        self.nvr_combo.bind("<<ComboboxSelected>>", self.on_nvr_selected)

        ttk.Label(control_frame, text="Камера:", background="#2e2e2e", foreground="white").pack(side="left", padx=10)
        self.camera_combo = ttk.Combobox(control_frame, textvariable=self.selected_camera, values=[], state="disabled", width=25)
        self.camera_combo.pack(side="left", padx=5)
        self.camera_combo.bind("<<ComboboxSelected>>", self.on_camera_selected)

        ttk.Label(control_frame, text="Отображать камер:", background="#2e2e2e", foreground="white").pack(side="left", padx=10)
        self.grid_combo = ttk.Combobox(control_frame, textvariable=self.selected_grid, values=[str(x) for x in GRID_SIZES], state="readonly", width=5)
        self.grid_combo.pack(side="left", padx=5)
        self.grid_combo.bind("<<ComboboxSelected>>", self.on_grid_change)

        # Область видео
        self.video_container = tk.Frame(root, bg="black")
        self.video_container.pack(fill="both", expand=True)
        self.video_container.bind("<Configure>", self.on_video_area_resize)

        self.root.after(100, self.on_nvr_selected)

    def on_nvr_selected(self, event=None):
        self.camera_combo.config(state="disabled")
        self.camera_combo.set("")
        threading.Thread(target=self._fetch_cameras, args=(self.selected_nvr.get(),), daemon=True).start()

    def _fetch_cameras(self, nvr_ip):
        camera_list = get_camera_list_from_nvr(nvr_ip)
        self.root.after(0, self._update_camera_list, camera_list)

    def _update_camera_list(self, camera_list):
        if camera_list:
            self.camera_list = camera_list
            display_names = [f"{cam['name']} ({cam['ip']})" for cam in camera_list]
            self.camera_combo["values"] = display_names
            if display_names:
                self.camera_combo.set(display_names[0])
                self.camera_combo.config(state="readonly")
            self.redraw_grid()
        else:
            self.camera_list = []
            self.camera_combo["values"] = []
            self.camera_combo.set("")
            self.camera_combo.config(state="disabled")
            messagebox.showwarning("Внимание", f"Не удалось получить список камер с {self.selected_nvr.get()}")
            self.redraw_grid()

    def on_camera_selected(self, event=None):
        self.redraw_grid()

    def on_grid_change(self, event=None):
        self.redraw_grid()

    def on_video_area_resize(self, event=None):
        if event and (event.width <= 1 or event.height <= 1):
            return
        self.redraw_grid()

    def redraw_grid(self):
        if self._redraw_scheduled:
            return
        self._redraw_scheduled = True
        self.root.after(100, self._do_redraw_grid)

    def _do_redraw_grid(self):
        self._redraw_scheduled = False

        # 🔥 КЛЮЧЕВОЕ ИЗМЕНЕНИЕ: не останавливаем плееры сразу!
        old_players = self.vlc_players.copy()
        self.vlc_players.clear()

        # Уничтожаем фреймы → HWND исчезает → VLC теряет вывод
        for widget in self.video_container.winfo_children():
            widget.destroy()

        # Откладываем освобождение ресурсов VLC на 1.5 сек в фоне
        if old_players:
            def delayed_cleanup():
                time.sleep(1.5)  # даём время на отвязку от HWND
                for player in old_players:
                    player.release_later()
            threading.Thread(target=delayed_cleanup, daemon=True).start()

        # --- Далее — создание новых фреймов и плееров ---
        if not self.camera_list:
            return

        try:
            count = int(self.selected_grid.get())
        except ValueError:
            count = 1

        start_index = 0
        selected_name = self.selected_camera.get()
        if selected_name:
            for i, cam in enumerate(self.camera_list):
                if selected_name == f"{cam['name']} ({cam['ip']})":
                    start_index = i
                    break

        cams_to_show = []
        for i in range(count):
            idx = (start_index + i) % len(self.camera_list)
            cams_to_show.append(self.camera_list[idx])

        if count == 1:
            rows, cols = 1, 1
        elif count <= 4:
            rows, cols = 2, 2
        elif count <= 9:
            rows, cols = 3, 3
        elif count <= 16:
            rows, cols = 4, 4
        else:
            rows, cols = 6, 6

        total_width = self.video_container.winfo_width()
        total_height = self.video_container.winfo_height()

        if total_width <= 1 or total_height <= 1:
            self.root.after(100, self._do_redraw_grid)
            return

        cell_width = total_width // cols
        cell_height = total_height // rows

        for idx, cam in enumerate(cams_to_show):
            r = idx // cols
            c = idx % cols
            x = c * cell_width
            y = r * cell_height

            frame = tk.Frame(self.video_container, bg="black", highlightbackground="gray", highlightthickness=1)
            frame.place(x=x, y=y, width=cell_width, height=cell_height)

            self.root.after(10, self._create_player, frame, cam)

    def _create_player(self, frame, cam):
        try:
            if not frame.winfo_exists():
                return
        except:
            return

        rtsp_url = get_rtsp_url_direct(cam['ip'], stream_type="secondary")
        try:
            player = VlcVideoPlayer(rtsp_url, frame)
            self.vlc_players.append(player)
        except Exception as e:
            print(f"❌ Не удалось создать плеер для {cam['ip']}: {e}")
            try:
                frame.destroy()
            except:
                pass


if __name__ == "__main__":
    root = tk.Tk()
    app = MultiCameraViewerApp(root)
    root.mainloop()