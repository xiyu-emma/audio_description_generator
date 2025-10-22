# main.py (單一環境最終整合版 - 已修正錯誤處理)

import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, simpledialog, messagebox
import subprocess
import sys
import os
import threading
import time
import traceback

# --- 語音功能 ---
try:
    from voice_interface import speak, voice_input, VoiceCommands, audio
    VOICE_ENABLED = True
except ImportError:
    print("[警告] voice_interface.py 未找到或導入失敗，語音功能將被禁用。")
    VOICE_ENABLED = False
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
status_bar = None # 將 status_bar 也設為全域

# 暫存資訊
_last_selected_image_path = None
_current_image_tk = None
_video_cap = None
_video_after_job = None
_current_video_path = None

# --- GUI 輔助函式 ---

def update_gui_safe(widget, text):
    """安全地從背景執行緒更新 ScrolledText 元件"""
    # 加入檢查: 確保 widget 和 app_window 仍然存在
    if widget and app_window and app_window.winfo_exists() and widget.winfo_exists():
        try:
            widget.config(state=tk.NORMAL)
            widget.insert(tk.END, text + "\n")
            widget.see(tk.END) # 自動捲動
            widget.config(state=tk.DISABLED)
        except tk.TclError as e:
            # 捕獲特定錯誤，例如視窗關閉時的 'invalid command name'
            print(f"更新 GUI 時發生 TclError (可能視窗已關閉): {e}")
        except Exception as e:
            print(f"更新 GUI 時發生未知錯誤: {e}")

def update_status_safe(text):
    """安全地更新狀態列文字"""
    # 加入檢查: 確保 status_label_var 和 app_window 仍然存在
    if status_label_var and app_window and app_window.winfo_exists():
        try:
            status_label_var.set(text)
        except tk.TclError as e:
            print(f"更新狀態列時發生 TclError (可能視窗已關閉): {e}")
        except Exception as e:
            print(f"更新狀態列時發生未知錯誤: {e}")

# 簡易工具提示類別
class ToolTip:
    # (Tooltip 類別程式碼已加入 winfo_exists 檢查)
    def __init__(self, widget, text: str, delay: int = 500):
        self.widget = widget
        self.text = text
        self.delay = delay
        self.tipwindow = None
        self._id = None
        widget.bind("<Enter>", self._enter)
        widget.bind("<Leave>", self._leave)
        widget.bind("<ButtonPress>", self._leave)

    def _enter(self, event=None): self._schedule()
    def _leave(self, event=None): self._unschedule(); self._hidetip()

    def _schedule(self):
        self._unschedule()
        # 加入檢查
        if self.widget.winfo_exists():
            self._id = self.widget.after(self.delay, self._showtip)

    def _unschedule(self):
        if self._id:
            try:
                # 加入檢查
                if self.widget.winfo_exists():
                    self.widget.after_cancel(self._id)
            except tk.TclError: pass
            self._id = None

    def _showtip(self, event=None):
        # 加入檢查
        if self.tipwindow or not self.text or not self.widget.winfo_exists():
            return
        try: bbox = self.widget.bbox("insert") if hasattr(self.widget, "bbox") else None
        except Exception: bbox = None
        x, y = (0, 0) if not bbox else (bbox[0], bbox[1] + bbox[3])
        x += self.widget.winfo_rootx() + 20
        y += self.widget.winfo_rooty() + 20

        try:
             self.tipwindow = tw = tk.Toplevel(self.widget)
             try: tw.wm_overrideredirect(1)
             except Exception: pass
             tw.configure(bg="#111827")
             label = tk.Label(tw, text=self.text, justify=tk.LEFT, background="#111827",
                              foreground="white", relief=tk.SOLID, borderwidth=1,
                              font=("Helvetica", 9), padx=6, pady=3)
             label.pack()
             # 在設定 geometry 前也檢查 widget 是否存在
             if self.widget.winfo_exists():
                 tw.wm_geometry(f"+{x}+{y}")
        except Exception as e:
            print(f"ToolTip _showtip error: {e}")
            self._hidetip() # 出錯時嘗試隱藏

    def _hidetip(self):
        tw = self.tipwindow
        self.tipwindow = None
        if tw:
            try:
                # 加入檢查
                if tw.winfo_exists():
                    tw.destroy()
            except Exception: pass

# --- 顯示圖片和文字 ---
def show_image_and_text(image_path: str, narration_text: str):
    """在 GUI 中顯示圖片預覽和生成的口述影像文字"""
    global _current_image_tk
    # 加入檢查
    if not app_window or not app_window.winfo_exists(): return

    try:
        from PIL import Image, ImageTk
    except ImportError:
        update_gui_safe(result_text_widget, "[警告] 需要 Pillow 函式庫來顯示圖片預覽 (pip install Pillow)")
        if narration_output_widget and narration_output_widget.winfo_exists(): # 檢查元件是否存在
             try:
                 narration_output_widget.config(state=tk.NORMAL)
                 narration_output_widget.delete('1.0', tk.END)
                 narration_output_widget.insert(tk.END, narration_text.strip() + "\n")
                 narration_output_widget.config(state=tk.DISABLED)
             except tk.TclError: pass
        return

    # 加入檢查
    if not image_preview_label or not image_preview_label.winfo_exists() or \
       not narration_output_widget or not narration_output_widget.winfo_exists():
        return

    # 顯示圖片
    try:
        img = Image.open(image_path)
        max_w, max_h = 480, 360
        img.thumbnail((max_w, max_h), Image.LANCZOS)
        _current_image_tk = ImageTk.PhotoImage(img)
        image_preview_label.config(image=_current_image_tk)
        image_preview_label.image = _current_image_tk
    except Exception as e:
        update_gui_safe(result_text_widget, f"[警告] 顯示圖片預覽失敗: {e}")
        try:
             image_preview_label.config(image='')
             image_preview_label.image = None
        except tk.TclError: pass

    # 顯示文字
    try:
        narration_output_widget.config(state=tk.NORMAL)
        narration_output_widget.delete('1.0', tk.END)
        narration_output_widget.insert(tk.END, narration_text.strip() + "\n")
        narration_output_widget.config(state=tk.DISABLED)
    except tk.TclError: pass

# --- 影片播放相關函式 ---
def stop_video_playback():
    """停止 UI 中的影片預覽"""
    global _video_cap, _video_after_job
    if _video_after_job and app_window and app_window.winfo_exists():
        try: app_window.after_cancel(_video_after_job)
        except tk.TclError: pass
        _video_after_job = None
    if _video_cap is not None:
        try: _video_cap.release()
        except Exception: pass
        _video_cap = None
    if video_preview_label and video_preview_label.winfo_exists():
        try:
             video_preview_label.config(image='')
             video_preview_label.image = None
        except tk.TclError: pass

def _update_video_frame():
    """讀取並顯示下一幀影片"""
    global _video_cap, _video_after_job
    # 加入檢查
    if not app_window or not app_window.winfo_exists(): return

    try:
        import cv2
        from PIL import Image, ImageTk
    except ImportError:
        update_gui_safe(result_text_widget, "[警告] 需要 opencv-python 和 Pillow 才能預覽影片")
        stop_video_playback()
        return

    # 加入檢查
    if _video_cap is None or not video_preview_label or not video_preview_label.winfo_exists():
         return

    ret, frame = _video_cap.read()
    if not ret:
        stop_video_playback()
        return

    try:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        max_w, max_h = 640, 360
        img.thumbnail((max_w, max_h), Image.LANCZOS)
        tk_img = ImageTk.PhotoImage(img)
        video_preview_label.config(image=tk_img)
        video_preview_label.image = tk_img

        fps = _video_cap.get(cv2.CAP_PROP_FPS) or 24.0
        delay = int(1000 / max(1.0, fps))
        # 加入檢查
        if app_window and app_window.winfo_exists():
            _video_after_job = app_window.after(delay, _update_video_frame)
    except tk.TclError: # 捕獲視窗關閉的錯誤
        stop_video_playback()
    except Exception as e:
        print(f"更新影片幀時出錯: {e}")
        stop_video_playback()

def play_video_in_ui(video_path: str):
    """開始在 UI 中預覽影片"""
    global _video_cap, _current_video_path
    stop_video_playback()
    _current_video_path = video_path
    try: import cv2
    except ImportError:
        update_gui_safe(result_text_widget, "[警告] 需要 opencv-python 才能預覽影片")
        return

    _video_cap = cv2.VideoCapture(video_path)
    if not _video_cap or not _video_cap.isOpened():
        update_gui_safe(result_text_widget, f"[警告] 無法開啟影片檔案進行預覽：{video_path}")
        return

    print(f"開始預覽影片: {video_path}")
    _update_video_frame()

def open_video_external():
    """使用系統預設播放器開啟影片"""
    if not _current_video_path or not os.path.exists(_current_video_path):
        messagebox.showwarning("無法開啟", "沒有可開啟的影片檔案。請先生成影片。")
        return
    path = os.path.normpath(_current_video_path)
    try:
        print(f"嘗試開啟外部影片: {path}")
        if sys.platform.startswith('win'): os.startfile(path)
        elif sys.platform == 'darwin': subprocess.Popen(['open', path])
        else: subprocess.Popen(['xdg-open', path])
    except Exception as e:
        update_gui_safe(result_text_widget, f"[警告] 開啟外部播放器失敗：{e}")
        messagebox.showerror("開啟失敗", f"無法使用系統播放器開啟影片:\n{e}")

# --- 執行緒函式 ---
def run_script_in_thread(script_name: str, script_type: str, args: list):
    """在背景執行緒中執行腳本並將輸出傳回 GUI (已加入 winfo_exists 檢查)"""
    global _last_selected_image_path
    # 安全地更新狀態
    if app_window and app_window.winfo_exists():
        app_window.after(0, update_status_safe, f"正在執行 {script_type} 程序...")
        app_window.after(0, update_gui_safe, result_text_widget, f"\n--- 開始執行 {script_name} ---")
    if VOICE_ENABLED: speak(f"正在啟動，{script_type}口述影像生成程序")

    final_answer = f"[{script_type} 未返回明確答案]"
    final_video_path = None
    final_image_path = None
    capture_next_video_path = False

    process = None # 初始化 process 變數
    try:
        script_path = os.path.join(os.path.dirname(__file__), script_name)
        command = [sys.executable, script_path] + args
        print(f"執行指令: {' '.join(command)}")

        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding='utf-8', errors='replace', bufsize=1,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
        )

        # --- 即時讀取 stdout ---
        if process.stdout:
            for line in iter(process.stdout.readline, ''):
                print(line, end='')
                # 安全地更新 GUI
                if app_window and app_window.winfo_exists():
                    app_window.after(0, update_gui_safe, result_text_widget, line.strip())
                # --- 解析特定輸出標記 ---
                s_line = line.strip()
                if s_line.startswith("FINAL_ANSWER:"): final_answer = s_line.replace("FINAL_ANSWER:", "").strip()
                elif s_line.startswith("FINAL_VIDEO:"): final_video_path = s_line.replace("FINAL_VIDEO:", "").strip()
                elif s_line.startswith("FINAL_IMAGE:"): final_image_path = s_line.replace("FINAL_IMAGE:", "").strip()
                elif "最終影片已儲存為：" in s_line: capture_next_video_path = True
                elif capture_next_video_path and s_line: final_video_path = s_line; capture_next_video_path = False
            process.stdout.close()

        # --- 讀取 stderr ---
        stderr_output = ""
        if process.stderr:
            stderr_output = process.stderr.read()
            process.stderr.close()

        return_code = process.wait() # 等待程序結束

        # --- 處理結果 ---
        if return_code == 0:
            success_msg = f"--- {script_name} 執行成功 ---"
            print(success_msg)
            # 安全地更新 GUI
            if app_window and app_window.winfo_exists():
                app_window.after(0, update_gui_safe, result_text_widget, success_msg)
                app_window.after(0, update_status_safe, f"{script_type} 完成")
            if VOICE_ENABLED: speak(f"{script_type} 處理完成")

            # --- 根據腳本類型更新 UI (安全地) ---
            if script_name == 'generate_image_ad.py':
                img_to_show = final_image_path or _last_selected_image_path
                if img_to_show and final_answer != f"[{script_type} 未返回明確答案]":
                     if app_window and app_window.winfo_exists(): # 檢查
                         app_window.after(0, show_image_and_text, img_to_show, final_answer)
                else:
                     if app_window and app_window.winfo_exists(): # 檢查
                         app_window.after(0, update_gui_safe, result_text_widget, "[提示] 未找到圖片路徑或生成結果用於顯示。")

            elif script_name == 'generate_video_ad.py':
                if final_video_path and os.path.exists(final_video_path):
                    if app_window and app_window.winfo_exists(): # 檢查
                        app_window.after(0, play_video_in_ui, final_video_path)
                        app_window.after(0, update_gui_safe, result_text_widget, f"[提示] 影片已生成: {final_video_path}")
                else:
                    if app_window and app_window.winfo_exists(): # 檢查
                        app_window.after(0, update_gui_safe, result_text_widget, "[警告] 未找到生成的影片檔案路徑或檔案不存在。")

        else: # 子程序執行失敗
            error_msg_header = f"\n!!!!!!!!!! {script_name} 執行時發生嚴重錯誤 !!!!!!!!!!\n返回碼: {return_code}"
            error_details = stderr_output if stderr_output else "[無詳細錯誤輸出]"
            error_msg_stderr = f"\n--- 錯誤輸出 (stderr) ---\n{error_details}\n-------------------------"
            full_error_msg = error_msg_header + error_msg_stderr
            print(full_error_msg)
            # 安全地更新 GUI
            if app_window and app_window.winfo_exists():
                app_window.after(0, update_gui_safe, result_text_widget, full_error_msg)
                app_window.after(0, update_status_safe, f"{script_type} 執行失敗")
            if VOICE_ENABLED: speak(f"啟動 {script_type} 處理程序時發生錯誤"); audio.beep_error()

    except FileNotFoundError:
        error_msg = f"錯誤：找不到腳本檔案 '{script_name}' 或 Python 執行檔 '{sys.executable}'"
        print(error_msg)
        # 安全地更新 GUI
        if app_window and app_window.winfo_exists():
             app_window.after(0, update_gui_safe, result_text_widget, error_msg)
             app_window.after(0, update_status_safe, f"{script_type} 失敗 (找不到檔案)")
        if VOICE_ENABLED: speak(f"啟動{script_type}失敗，找不到檔案"); audio.beep_error()
    except Exception as e:
        error_msg = f"執行 {script_name} 時發生未預期的錯誤: {e}\n{traceback.format_exc()}"
        print(error_msg)
        # 安全地更新 GUI
        if app_window and app_window.winfo_exists():
             app_window.after(0, update_gui_safe, result_text_widget, error_msg)
             app_window.after(0, update_status_safe, f"{script_type} 失敗 (未知錯誤)")
        if VOICE_ENABLED: speak(f"啟動{script_type}時發生未知錯誤"); audio.beep_error()
    finally:
        # 確保 process 被關閉 (如果它被成功創建的話)
        if process and process.poll() is None: # 如果 process 還在運行
            try:
                process.terminate() # 嘗試終止
                process.wait(timeout=1) # 等待一小段時間
            except Exception as e:
                print(f"嘗試終止子程序時出錯: {e}")
            finally: # 無論如何都嘗試 kill
                 if process.poll() is None:
                     process.kill()

        # 安全地更新 GUI 狀態
        if app_window and app_window.winfo_exists():
            app_window.after(100, enable_buttons)
            app_window.after(0, set_busy, False)

def enable_buttons():
    """重新啟用主按鈕 (加入檢查)"""
    try:
        # 檢查元件是否存在
        if image_button and image_button.winfo_exists(): image_button.config(state=tk.NORMAL)
        if video_button and video_button.winfo_exists(): video_button.config(state=tk.NORMAL)
    except tk.TclError:
        pass # 視窗可能已關閉

def set_busy(is_busy: bool):
    """設定 GUI 為忙碌或空閒狀態 (加入檢查)"""
    global app_window, progress_bar, status_bar
    # 加入檢查
    if not app_window or not app_window.winfo_exists() or progress_bar is None: return

    try:
        if is_busy:
            # 確保 status_bar 存在
            if status_bar and status_bar.winfo_exists():
                progress_bar.pack(side=tk.BOTTOM, fill=tk.X, before=status_bar)
            else: # 備案：直接 pack
                 progress_bar.pack(side=tk.BOTTOM, fill=tk.X)
            try: progress_bar.start(10)
            except tk.TclError: pass
            app_window.config(cursor='watch')
        else:
            try: progress_bar.stop()
            except tk.TclError: pass
            progress_bar.pack_forget()
            app_window.config(cursor='')
    except tk.TclError:
        pass # 視窗可能已關閉

# --- 啟動流程 ---
def start_image_analysis():
    # (此函式基本不變，已包含相關檢查)
    global _last_selected_image_path
    file_path = filedialog.askopenfilename(title="請選擇一張圖片", filetypes=[("Image Files", "*.png *.jpg *.jpeg *.webp *.bmp")])
    if not file_path: return

    desc = simpledialog.askstring("圖片描述", "請輸入這張圖片的描述或重點：", parent=app_window)
    if desc is None: return
    if not desc.strip():
        messagebox.showwarning("輸入錯誤", "圖片描述不能為空。", parent=app_window)
        return

    model_dir = os.path.join(".", "models", "Llama-3.2-11B-Vision-Instruct")
    if not os.path.isdir(model_dir):
         messagebox.showerror("缺少模型", f"在相對路徑 '{model_dir}' 下找不到 Llama 模型資料夾。\n請確認模型已下載並放置在正確位置。", parent=app_window)
         return

    _last_selected_image_path = file_path

    # 清理舊輸出 (加入檢查)
    if result_text_widget and result_text_widget.winfo_exists():
        try: result_text_widget.config(state=tk.NORMAL); result_text_widget.delete('1.0', tk.END); result_text_widget.config(state=tk.DISABLED)
        except tk.TclError: pass
    if narration_output_widget and narration_output_widget.winfo_exists():
        try: narration_output_widget.config(state=tk.NORMAL); narration_output_widget.delete('1.0', tk.END); narration_output_widget.config(state=tk.DISABLED)
        except tk.TclError: pass
    if image_preview_label and image_preview_label.winfo_exists():
        try: image_preview_label.config(image=''); image_preview_label.image = None
        except tk.TclError: pass
    stop_video_playback()

    # 禁用按鈕並設定忙碌
    try:
        if image_button and image_button.winfo_exists(): image_button.config(state=tk.DISABLED)
        if video_button and video_button.winfo_exists(): video_button.config(state=tk.DISABLED)
    except tk.TclError: pass
    set_busy(True)

    args = ["--model_path", model_dir, "--image_file", file_path, "--desc", desc]
    thread = threading.Thread(target=run_script_in_thread, args=('generate_image_ad.py', '圖像', args), daemon=True)
    thread.start()

def start_video_analysis():
    # (此函式基本不變，已包含相關檢查)
    # 清理舊輸出 (加入檢查)
    if result_text_widget and result_text_widget.winfo_exists():
        try: result_text_widget.config(state=tk.NORMAL); result_text_widget.delete('1.0', tk.END); result_text_widget.config(state=tk.DISABLED)
        except tk.TclError: pass
    if narration_output_widget and narration_output_widget.winfo_exists():
        try: narration_output_widget.config(state=tk.NORMAL); narration_output_widget.delete('1.0', tk.END); narration_output_widget.config(state=tk.DISABLED)
        except tk.TclError: pass
    if image_preview_label and image_preview_label.winfo_exists():
        try: image_preview_label.config(image=''); image_preview_label.image = None
        except tk.TclError: pass
    stop_video_playback()

    # 禁用按鈕並設定忙碌
    try:
        if image_button and image_button.winfo_exists(): image_button.config(state=tk.DISABLED)
        if video_button and video_button.winfo_exists(): video_button.config(state=tk.DISABLED)
    except tk.TclError: pass
    set_busy(True)

    args = []
    thread = threading.Thread(target=run_script_in_thread, args=('generate_video_ad.py', '影片', args), daemon=True)
    thread.start()

# --- 語音互動迴圈 ---
def voice_interaction_loop():
    # (加入 app_window 檢查)
    if not VOICE_ENABLED: return
    time.sleep(1.5)
    # 檢查視窗是否存在
    if not app_window or not app_window.winfo_exists(): return
    speak("歡迎使用口述影像生成系統")
    session_active = True
    while session_active:
        # 再次檢查視窗
        if not app_window or not app_window.winfo_exists(): break
        prompt = "請說出指令：生成圖像、生成影片，或 結束"
        command = voice_input(prompt)
        if not command: continue
        # 再次檢查視窗
        if not app_window or not app_window.winfo_exists(): break

        parsed = VoiceCommands.parse(command)
        if parsed == "image": app_window.after(0, start_image_analysis)
        elif parsed == "video": app_window.after(0, start_video_analysis)
        elif parsed == "exit":
            speak("感謝您的使用，系統即將關閉")
            if VOICE_ENABLED: audio.beep_success()
            if app_window and app_window.winfo_exists(): app_window.destroy()
            session_active = False
        else:
            speak("無法辨識指令，請重新說一次")
            if VOICE_ENABLED: audio.beep_error()

# --- GUI 建立 ---
def create_gui():
    # (此函式已包含 status_bar 的 global 宣告，無需修改 Tooltip 之外的部分)
    global result_text_widget, status_label_var, app_window
    global image_button, video_button, progress_bar
    global image_preview_label, narration_output_widget, video_preview_label
    global status_bar # 已加入

    root = tk.Tk()
    app_window = root
    root.title("口述影像生成系統")
    root.geometry("1000x780")
    root.minsize(900, 680)

    # --- 主題與色彩 ---
    style = ttk.Style()
    try: style.theme_use("clam")
    except Exception: pass
    ACCENT, ACCENT_HOVER, BG, TEXT, SUBTEXT, BORDER = "#4F46E5", "#4338CA", "#F8FAFC", "#111827", "#6B7280", "#E5E7EB"
    try: root.configure(background=BG)
    except tk.TclError: pass

    # --- 設定元件樣式 ---
    style.configure("TFrame", background=BG)
    style.configure("TLabel", background=BG, foreground=TEXT, font=("Helvetica", 10))
    style.configure("Header.TLabel", background=BG, foreground=TEXT, font=("Helvetica", 20, "bold"))
    style.configure("SubHeader.TLabel", background=BG, foreground=SUBTEXT, font=("Helvetica", 11))
    style.configure("TLabelFrame", background=BG, foreground=TEXT, bordercolor=BORDER, relief=tk.SOLID, borderwidth=1)
    style.configure("TLabelFrame.Label", background=BG, foreground=TEXT, font=("Helvetica", 11, "bold"))
    style.configure("TButton", font=("Helvetica", 12), padding=(12, 10), borderwidth=1)
    style.configure("Primary.TButton", background=ACCENT, foreground="white", relief=tk.FLAT)
    style.map("Primary.TButton", background=[("active", ACCENT_HOVER), ("disabled", "#9CA3AF")], foreground=[("disabled", "#E5E7EB")])
    style.configure("Secondary.TButton", background="#E5E7EB", foreground=TEXT, relief=tk.FLAT)
    style.map("Secondary.TButton", background=[("active", "#D1D5DB")])
    style.configure("Horizontal.TProgressbar", troughcolor=BORDER, background=ACCENT)
    style.configure("Status.TLabel", background="#1F2937", foreground="#D1D5DB", font=("Consolas", 9))

    # --- 主要容器 ---
    main_frame = ttk.Frame(root, padding=20)
    main_frame.pack(expand=True, fill="both")

    # --- 標題區 ---
    header_label = ttk.Label(main_frame, text="口述影像生成系統", style="Header.TLabel")
    header_label.pack(anchor="w")
    subheader_label = ttk.Label(main_frame, text="為圖像與影片生成高品質的口述影像旁白", style="SubHeader.TLabel")
    subheader_label.pack(anchor="w", pady=(0, 15))

    # --- 功能按鈕區 ---
    btn_frame = ttk.Frame(main_frame)
    btn_frame.pack(fill="x", pady=(5, 10))
    image_button = ttk.Button(btn_frame, text="🖼️ 生成圖像口述影像", command=start_image_analysis, style="Primary.TButton", width=30)
    image_button.pack(side="left", expand=True, fill="x", padx=(0, 8))
    video_button = ttk.Button(btn_frame, text="🎬 生成影片口述影像", command=start_video_analysis, style="Primary.TButton", width=30)
    video_button.pack(side="left", expand=True, fill="x", padx=(8, 0))

    # --- 工具提示 ---
    try:
        ToolTip(image_button, "點擊以上傳單張圖片並輸入描述，\n使用 Llama 模型生成口述影像。")
        ToolTip(video_button, "點擊以選擇影片檔案，\n使用 Gemini 模型自動生成口述影像。")
    except Exception as e: print(f"無法建立工具提示: {e}")

    # --- 視覺輸出區 ---
    output_area_frame = ttk.Frame(main_frame)
    output_area_frame.pack(expand=True, fill="both", pady=10)
    image_output_frame = ttk.LabelFrame(output_area_frame, text="圖像結果預覽", labelanchor="n", padding=10)
    image_output_frame.pack(side="left", expand=True, fill="both", padx=(0, 10))
    image_preview_label = ttk.Label(image_output_frame, text="[此處顯示圖片預覽]", anchor=tk.CENTER, background=BORDER) # 用BORDER色
    image_preview_label.pack(fill="x", pady=(5, 10))
    ttk.Label(image_output_frame, text="生成的口述影像:", font=("Helvetica", 10, "bold")).pack(anchor="w", pady=(5,2))
    narration_output_widget = scrolledtext.ScrolledText(image_output_frame, wrap=tk.WORD, height=6, state=tk.DISABLED, font=("Helvetica", 10), relief=tk.SOLID, borderwidth=1, bd=1)
    narration_output_widget.pack(expand=True, fill="both")

    video_output_frame = ttk.LabelFrame(output_area_frame, text="影片結果預覽", labelanchor="n", padding=10)
    video_output_frame.pack(side="left", expand=True, fill="both", padx=(10, 0))
    video_preview_label = ttk.Label(video_output_frame, text="[此處顯示影片預覽]", anchor=tk.CENTER, background=BORDER) # 用BORDER色
    video_preview_label.pack(fill="x", pady=(5, 10))
    open_external_btn = ttk.Button(video_output_frame, text="▶️ 在系統播放器中開啟", command=open_video_external, style="Secondary.TButton")
    open_external_btn.pack(pady=(5, 5))
    try: ToolTip(open_external_btn, "使用系統預設播放器開啟生成的影片檔案")
    except Exception: pass

    # --- 執行日誌輸出區 ---
    result_frame = ttk.LabelFrame(main_frame, text="執行日誌", labelanchor="n", padding=10)
    result_frame.pack(expand=True, fill="both", pady=(10, 0))
    result_text_widget = scrolledtext.ScrolledText(result_frame, wrap=tk.WORD, height=10, state=tk.DISABLED, font=("Consolas", 9), relief=tk.SOLID, borderwidth=1, bd=1, background="#F9FAFB", foreground="#374151")
    result_text_widget.pack(expand=True, fill="both")

    # --- 狀態列與進度列 ---
    status_label_var = tk.StringVar(value="準備就緒")
    status_bar = ttk.Label(root, textvariable=status_label_var, anchor=tk.W, padding=(8, 5), style="Status.TLabel")
    status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    progress_bar = ttk.Progressbar(root, mode="indeterminate", style="Horizontal.TProgressbar")

    return root

# --- 程式主進入點 ---
if __name__ == "__main__":
    app_window = create_gui()

    if VOICE_ENABLED:
        voice_thread = threading.Thread(target=voice_interaction_loop, daemon=True)
        voice_thread.start()
    else:
        update_status_safe("語音功能未啟用")

    # 綁定關閉視窗事件
    app_window.protocol("WM_DELETE_WINDOW", lambda: (stop_video_playback(), app_window.destroy()))

    app_window.mainloop()

    # 清理資源
    stop_video_playback()
    print("應用程式已關閉。")