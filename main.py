# main.py (單一環境版本，兩個按鈕：生成圖像口述影像 / 生成影片口述影像)

import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, simpledialog, messagebox
import subprocess
import sys
import os
import threading
import time
import traceback

# --- 語音功能 ---
# 保持 voice_interface.py 的 import，如果檔案存在
try:
    from voice_interface import speak, voice_input, VoiceCommands, audio
    VOICE_ENABLED = True
except ImportError:
    print("[警告] voice_interface.py 未找到或導入失敗，語音功能將被禁用。")
    VOICE_ENABLED = False
    # 定義假的函式以避免錯誤
    def speak(text, **kwargs): print(f"[語音模擬]: {text}")
    def voice_input(prompt, **kwargs): print(f"[語音模擬] 提示: {prompt}"); return None
    class DummyAudio:
        def beep_error(self): pass
        def beep_success(self): pass
    audio = DummyAudio()
    class DummyVoiceCommands:
        def parse(self, text): return text
    VoiceCommands = DummyVoiceCommands()
# --- 語音功能結束 ---

# --- 全域變數 ---
result_text_widget = None
status_label_var = None
app_window = None
image_button = None
video_button = None
image_preview_label = None
narration_output_widget = None
video_preview_label = None
progress_bar = None

# 圖像/影片暫存資訊
_last_selected_image_path = None
_current_image_tk = None
_video_cap = None
_video_after_job = None
_current_video_path = None

# --- GUI 輔助函式 ---
def update_gui_safe(widget, text):
    if widget and app_window:
        try:
            widget.config(state=tk.NORMAL)
            widget.insert(tk.END, text + "\n")
            widget.see(tk.END)
            widget.config(state=tk.DISABLED)
        except tk.TclError:
            pass

def update_status_safe(text):
    if status_label_var and app_window:
        try:
            status_label_var.set(text)
        except tk.TclError:
            pass

# 簡易工具提示類別，為按鈕提供滑鼠懸停說明
class ToolTip:
    def __init__(self, widget, text: str, delay: int = 500):
        self.widget = widget
        self.text = text
        self.delay = delay
        self.tipwindow = None
        self._id = None
        widget.bind("<Enter>", self._enter)
        widget.bind("<Leave>", self._leave)
        widget.bind("<ButtonPress>", self._leave)

    def _enter(self, event=None):
        self._schedule()

    def _leave(self, event=None):
        self._unschedule()
        self._hidetip()

    def _schedule(self):
        self._unschedule()
        self._id = self.widget.after(self.delay, self._showtip)

    def _unschedule(self):
        if self._id:
            try:
                self.widget.after_cancel(self._id)
            except tk.TclError:
                pass
            self._id = None

    def _showtip(self, event=None):
        if self.tipwindow or not self.text:
            return
        try:
            bbox = self.widget.bbox("insert") if hasattr(self.widget, "bbox") else None
        except Exception:
            bbox = None
        if bbox:
            x, y, cx, cy = bbox
        else:
            x, y, cx, cy = 0, 0, 0, 0
        x = x + self.widget.winfo_rootx() + 20
        y = y + cy + self.widget.winfo_rooty() + 20
        self.tipwindow = tw = tk.Toplevel(self.widget)
        try:
            tw.wm_overrideredirect(1)
        except Exception:
            pass
        tw.configure(bg="#111827")
        label = tk.Label(
            tw,
            text=self.text,
            justify=tk.LEFT,
            background="#111827",
            foreground="white",
            relief=tk.SOLID,
            borderwidth=1,
            font=("Helvetica", 9),
            padx=6,
            pady=3,
        )
        label.pack()
        tw.wm_geometry("+%d+%d" % (x, y))

    def _hidetip(self):
        tw = self.tipwindow
        self.tipwindow = None
        if tw:
            try:
                tw.destroy()
            except Exception:
                pass

# 視覺輸出輔助
def show_image_and_text(image_path: str, narration_text: str):
    global _current_image_tk
    try:
        from PIL import Image, ImageTk
    except Exception as e:
        update_gui_safe(result_text_widget, f"[警告] 無法載入 PIL 以顯示圖片：{e}")
        return

    if not image_preview_label or not narration_output_widget:
        return
    try:
        img = Image.open(image_path)
        max_w, max_h = 480, 360
        img.thumbnail((max_w, max_h))
        _current_image_tk = ImageTk.PhotoImage(img)
        image_preview_label.config(image=_current_image_tk)
        image_preview_label.image = _current_image_tk
    except Exception as e:
        update_gui_safe(result_text_widget, f"[警告] 顯示圖片失敗：{e}")

    try:
        narration_output_widget.config(state=tk.NORMAL)
        narration_output_widget.delete('1.0', tk.END)
        narration_output_widget.insert(tk.END, narration_text.strip() + "\n")
        narration_output_widget.config(state=tk.DISABLED)
    except tk.TclError:
        pass

# 影片播放輔助（在 UI 中預覽畫面）
def stop_video_playback():
    global _video_cap, _video_after_job
    if _video_after_job and app_window:
        try:
            app_window.after_cancel(_video_after_job)
        except tk.TclError:
            pass
        _video_after_job = None
    if _video_cap is not None:
        try:
            _video_cap.release()
        except Exception:
            pass
        _video_cap = None


def _update_video_frame():
    global _video_cap, _video_after_job
    try:
        import cv2
        from PIL import Image, ImageTk
    except Exception as e:
        update_gui_safe(result_text_widget, f"[警告] 無法載入顯示影片所需套件：{e}")
        return

    if _video_cap is None or not video_preview_label:
        return

    ret, frame = _video_cap.read()
    if not ret:
        # 結束或重播
        stop_video_playback()
        return

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    max_w, max_h = 640, 360
    img.thumbnail((max_w, max_h))
    tk_img = ImageTk.PhotoImage(img)
    video_preview_label.config(image=tk_img)
    video_preview_label.image = tk_img

    # 使用影片 FPS 控制更新
    fps = _video_cap.get(5) or 24.0
    delay = int(1000 / max(1.0, fps))
    _video_after_job = app_window.after(delay, _update_video_frame)


def play_video_in_ui(video_path: str):
    global _video_cap, _current_video_path
    stop_video_playback()
    _current_video_path = video_path
    try:
        import cv2
    except Exception as e:
        update_gui_safe(result_text_widget, f"[警告] 無法載入 OpenCV 以顯示影片：{e}")
        return

    _video_cap = cv2.VideoCapture(video_path)
    if not _video_cap or not _video_cap.isOpened():
        update_gui_safe(result_text_widget, f"[警告] 開啟影片失敗：{video_path}")
        return

    _update_video_frame()


def open_video_external():
    if not _current_video_path:
        return
    path = _current_video_path
    try:
        if sys.platform.startswith('win'):
            os.startfile(path)  # type: ignore
        elif sys.platform == 'darwin':
            subprocess.Popen(['open', path])
        else:
            subprocess.Popen(['xdg-open', path])
    except Exception as e:
        update_gui_safe(result_text_widget, f"[警告] 開啟外部播放器失敗：{e}")

# --- 執行緒函式 ---
def run_script_in_thread(script_name: str, script_type: str, args: list):
    global _last_selected_image_path
    update_status_safe(f"正在執行 {script_type} 程序...")
    update_gui_safe(result_text_widget, f"\n--- 開始執行 {script_name} ---")
    speak(f"正在啟動，{script_type}口述影像生成程序")

    try:
        script_path = os.path.join(os.path.dirname(__file__), script_name)
        command = [sys.executable, script_path] + args
        print(f"執行指令: {' '.join(command)}")

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding='utf-8', errors='replace',
            bufsize=1, creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
        )

        final_answer = f"[{script_type} 未返回明確答案]"
        final_video_path = None
        final_image_path = None
        capture_next_video_path = False

        # 即時讀取 stdout
        if process.stdout:
            for line in iter(process.stdout.readline, ''):
                print(line, end='')
                if app_window:
                    app_window.after(0, update_gui_safe, result_text_widget, line.strip())
                s = line.strip()
                if s.startswith("FINAL_ANSWER:"):
                    final_answer = s.replace("FINAL_ANSWER:", "").strip()
                elif s.startswith("FINAL_VIDEO:"):
                    final_video_path = s.replace("FINAL_VIDEO:", "").strip()
                elif s.startswith("FINAL_IMAGE:"):
                    final_image_path = s.replace("FINAL_IMAGE:", "").strip()
                elif "最終影片已儲存為：" in s:
                    capture_next_video_path = True
                elif capture_next_video_path and s:
                    final_video_path = s
                    capture_next_video_path = False
            process.stdout.close()

        stderr_output = ""
        if process.stderr:
            stderr_output = process.stderr.read()
            process.stderr.close()

        return_code = process.wait()

        if return_code == 0:
            success_msg = f"--- {script_name} 執行成功 ---"
            print(success_msg)
            if app_window:
                app_window.after(0, update_gui_safe, result_text_widget, success_msg)
                app_window.after(0, update_status_safe, f"{script_type} 完成")
            speak(f"{script_type} 處理完成")

            if script_name == 'generate_image_ad.py':
                img_path = final_image_path or _last_selected_image_path
                if img_path and final_answer:
                    app_window.after(0, show_image_and_text, img_path, final_answer)
            elif script_name == 'generate_video_ad.py':
                if final_video_path:
                    app_window.after(0, play_video_in_ui, final_video_path)

        else:
            error_msg_header = f"\n!!!!!!!!!! {script_name} 執行時發生嚴重錯誤 !!!!!!!!!!\n返回碼: {return_code}"
            error_msg_stderr = f"\n--- 錯誤輸出 (stderr) ---\n{stderr_output}\n-------------------------"
            print(error_msg_header)
            print(error_msg_stderr)
            if app_window:
                app_window.after(0, update_gui_safe, result_text_widget, error_msg_header + error_msg_stderr)
                app_window.after(0, update_status_safe, f"{script_type} 執行失敗")
            speak(f"啟動 {script_type} 處理程序時發生錯誤")
            if VOICE_ENABLED: audio.beep_error()

    except FileNotFoundError:
        error_msg = f"錯誤：找不到腳本檔案 '{script_name}' 或 Python 執行檔 '{sys.executable}'"
        print(error_msg)
        if app_window:
            app_window.after(0, update_gui_safe, result_text_widget, error_msg)
            app_window.after(0, update_status_safe, f"{script_type} 失敗 (找不到檔案)")
        speak(f"啟動{script_type}失敗，找不到檔案")
        if VOICE_ENABLED: audio.beep_error()
    except Exception as e:
        error_msg = f"執行 {script_name} 時發生未預期的錯誤: {e}\n{traceback.format_exc()}"
        print(error_msg)
        if app_window:
            app_window.after(0, update_gui_safe, result_text_widget, error_msg)
            app_window.after(0, update_status_safe, f"{script_type} 失敗 (未知錯誤)")
        speak(f"啟動{script_type}時發生未知錯誤")
        if VOICE_ENABLED: audio.beep_error()
    finally:
        if app_window:
            app_window.after(100, enable_buttons)
            app_window.after(0, set_busy, False)


def enable_buttons():
    try:
        if image_button: image_button.config(state=tk.NORMAL)
        if video_button: video_button.config(state=tk.NORMAL)
    except tk.TclError:
        pass


def set_busy(is_busy: bool):
    global app_window, progress_bar
    if not app_window or progress_bar is None:
        return
    try:
        if is_busy:
            # Place progress bar above status bar
            progress_bar.pack(side=tk.BOTTOM, fill=tk.X)
            try:
                progress_bar.start(10)
            except Exception:
                pass
            app_window.configure(cursor='watch')
        else:
            try:
                progress_bar.stop()
            except Exception:
                pass
            progress_bar.pack_forget()
            app_window.configure(cursor='')
    except tk.TclError:
        pass

# --- 啟動流程 ---
def start_image_analysis():
    global _last_selected_image_path
    # 1) 讓使用者上傳單張圖片
    file_path = filedialog.askopenfilename(
        title="請選擇一張圖片",
        filetypes=[("Image Files", "*.png *.jpg *.jpeg *.webp *.bmp")]
    )
    if not file_path:
        return

    # 2) 讓使用者輸入一段圖片的描述
    desc = simpledialog.askstring("圖片描述", "請輸入這張圖片的描述：", parent=app_window)
    if desc is None or not desc.strip():
        messagebox.showwarning("輸入錯誤", "請輸入圖片描述。")
        return

    model_dir = os.path.join(".", "models", "Llama-3.2-11B-Vision-Instruct")
    if not os.path.isdir(model_dir):
        messagebox.showerror("缺少模型", f"在相對路徑 '{model_dir}' 下找不到 Llama 模型資料夾。\n請確認模型已下載並放置在正確位置。")
        return

    _last_selected_image_path = file_path

    # 清空輸出區
    result_text_widget.config(state=tk.NORMAL)
    result_text_widget.delete('1.0', tk.END)
    result_text_widget.config(state=tk.DISABLED)
    if narration_output_widget:
        narration_output_widget.config(state=tk.NORMAL)
        narration_output_widget.delete('1.0', tk.END)
        narration_output_widget.config(state=tk.DISABLED)
    if image_preview_label:
        image_preview_label.config(image='')
        image_preview_label.image = None

    image_button.config(state=tk.DISABLED)
    video_button.config(state=tk.DISABLED)

    args = [
        "--model_path", model_dir,
        "--image_file", file_path,
        "--desc", desc
    ]

    set_busy(True)
    thread = threading.Thread(
        target=run_script_in_thread,
        args=('generate_image_ad.py', '圖像', args),
        daemon=True
    )
    thread.start()


def start_video_analysis():
    # 清空輸出區
    result_text_widget.config(state=tk.NORMAL)
    result_text_widget.delete('1.0', tk.END)
    result_text_widget.config(state=tk.DISABLED)

    # 停止任何現有影片播放
    stop_video_playback()
    if video_preview_label:
        video_preview_label.config(image='')
        video_preview_label.image = None

    image_button.config(state=tk.DISABLED)
    video_button.config(state=tk.DISABLED)

    args = []
    set_busy(True)
    thread = threading.Thread(
        target=run_script_in_thread,
        args=('generate_video_ad.py', '影片', args),
        daemon=True
    )
    thread.start()


# --- 語音互動迴圈 ---
def voice_interaction_loop():
    if not VOICE_ENABLED:
        print("語音功能未啟用。")
        return

    time.sleep(1.5)
    speak("歡迎使用口述影像生成系統")

    session_active = True
    while session_active:
        prompt = "請說出指令：生成圖像、生成影片，或 結束"
        command = voice_input(prompt)
        if not command:
            continue
        parsed = VoiceCommands.parse(command)
        if parsed == "image":
            if app_window:
                app_window.after(0, start_image_analysis)
        elif parsed == "video":
            if app_window:
                app_window.after(0, start_video_analysis)
        elif parsed == "exit":
            speak("感謝您的使用，系統即將關閉")
            if VOICE_ENABLED: audio.beep_success()
            if app_window:
                app_window.destroy()
            session_active = False
        else:
            speak("無法辨識指令，請重新說一次")
            if VOICE_ENABLED: audio.beep_error()


# --- GUI 建立 ---
def create_gui():
    global result_text_widget, status_label_var, app_window
    global image_button, video_button
    global image_preview_label, narration_output_widget, video_preview_label
    global progress_bar

    root = tk.Tk()
    app_window = root
    root.title("口述影像生成系統")
    root.geometry("1000x780")
    root.minsize(900, 680)

    # 主題與色彩配置
    style = ttk.Style()
    try:
        style.theme_use("clam")
    except Exception:
        pass

    ACCENT = "#4F46E5"
    ACCENT_HOVER = "#4338CA"
    BG = "#F8FAFC"
    TEXT = "#111827"
    SUBTEXT = "#6B7280"

    try:
        root.configure(background=BG)
    except tk.TclError:
        pass

    # 基礎樣式
    style.configure("TFrame", background=BG)
    style.configure("TLabel", background=BG, foreground=TEXT, font=("Helvetica", 10))
    style.configure("Header.TLabel", background=BG, foreground=TEXT, font=("Helvetica", 20, "bold"))
    style.configure("SubHeader.TLabel", background=BG, foreground=SUBTEXT, font=("Helvetica", 11))
    style.configure("TLabelFrame", background=BG, foreground=TEXT, padding=10)
    style.configure("TLabelFrame.Label", background=BG, foreground=TEXT)
    style.configure("TButton", font=("Helvetica", 12), padding=(12, 10))
    style.configure("Primary.TButton", background=ACCENT, foreground="white")
    style.map(
        "Primary.TButton",
        background=[("active", ACCENT_HOVER), ("disabled", "#9CA3AF")],
        foreground=[("disabled", "#E5E7EB")],
    )
    style.configure("Secondary.TButton", background="#E5E7EB", foreground=TEXT)
    style.map("Secondary.TButton", background=[("active", "#D1D5DB")])
    style.configure("Horizontal.TProgressbar", troughcolor="#E5E7EB", background=ACCENT)
    style.configure("Status.TLabel", background="#111827", foreground="white")

    # 主要容器
    main_frame = ttk.Frame(root, padding=16)
    main_frame.pack(expand=True, fill="both")

    # 標題區
    header_label = ttk.Label(main_frame, text="口述影像生成系統", style="Header.TLabel")
    header_label.pack(anchor="w")
    subheader_label = ttk.Label(main_frame, text="為視障者提供友善的圖像與影片旁白", style="SubHeader.TLabel")
    subheader_label.pack(anchor="w", pady=(0, 10))

    # 兩個主按鈕
    btn_frame = ttk.Frame(main_frame)
    btn_frame.pack(fill="x", pady=(4, 8))
    image_button = ttk.Button(btn_frame, text="🖼️ 生成圖像口述影像", command=start_image_analysis, style="Primary.TButton")
    image_button.pack(side="left", expand=True, fill="x", padx=(0, 6))
    video_button = ttk.Button(btn_frame, text="🎬 生成影片口述影像", command=start_video_analysis, style="Primary.TButton")
    video_button.pack(side="left", expand=True, fill="x", padx=(6, 0))

    # 工具提示
    try:
        ToolTip(image_button, "上傳一張圖片並生成口述影像")
        ToolTip(video_button, "偵測影片重點並生成口述影像")
    except Exception:
        pass

    # 圖像 + 文字輸出區
    preview_frame = ttk.LabelFrame(main_frame, text="圖像預覽與口述影像文字", labelanchor="nw")
    preview_frame.pack(fill="x", pady=10)

    content_frame = ttk.Frame(preview_frame)
    content_frame.pack(fill="x")

    image_preview_label = ttk.Label(content_frame)
    image_preview_label.pack(side="left", padx=10, pady=5)

    narration_output_widget = scrolledtext.ScrolledText(
        content_frame,
        wrap=tk.WORD,
        width=60,
        height=12,
        state=tk.DISABLED,
        font=("Helvetica", 10),
    )
    narration_output_widget.pack(side="left", expand=True, fill="both", padx=10, pady=5)

    # 影片預覽區
    video_frame = ttk.LabelFrame(main_frame, text="影片預覽", labelanchor="nw")
    video_frame.pack(fill="both", expand=False)

    video_preview_label = ttk.Label(video_frame)
    video_preview_label.pack(pady=8)

    open_external_btn = ttk.Button(
        video_frame,
        text="▶️ 在系統播放器中開啟影片",
        command=open_video_external,
        style="Secondary.TButton",
    )
    open_external_btn.pack(pady=(0, 8))

    try:
        ToolTip(open_external_btn, "使用系統預設播放器開啟所產生的影片")
    except Exception:
        pass

    # 執行結果輸出區
    result_frame = ttk.LabelFrame(main_frame, text="執行結果與日誌", labelanchor="nw")
    result_frame.pack(expand=True, fill="both", pady=(10, 0))
    result_text_widget = scrolledtext.ScrolledText(
        result_frame, wrap=tk.WORD, height=12, state=tk.DISABLED, font=("Consolas", 9)
    )
    result_text_widget.pack(expand=True, fill="both")

    # 狀態列與進度列
    status_label_var = tk.StringVar(value="準備就緒")
    status_bar = ttk.Label(root, textvariable=status_label_var, anchor=tk.W, padding=(6, 4), style="Status.TLabel")
    status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    progress_bar = ttk.Progressbar(root, mode="indeterminate")
    # 初始隱藏，由 set_busy 控制顯示與啟動

    return root


# --- 程式主進入點 ---
if __name__ == "__main__":
    app_window = create_gui()

    if VOICE_ENABLED:
        voice_thread = threading.Thread(target=voice_interaction_loop, daemon=True)
        voice_thread.start()
    else:
        update_status_safe("語音功能未啟用")

    app_window.mainloop()

    stop_video_playback()
    print("應用程式已關閉。")
