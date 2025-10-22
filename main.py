# main.py (單一環境最終整合版 - 已修正錯誤處理)

import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, simpledialog, messagebox
import subprocess
import sys
import os
import threading
import time
import traceback
import cv2
from PIL import Image, ImageTk
import tempfile

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
realtime_button = None
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
        if realtime_button and realtime_button.winfo_exists(): realtime_button.config(state=tk.NORMAL)
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
        if realtime_button and realtime_button.winfo_exists(): realtime_button.config(state=tk.DISABLED)
    except tk.TclError: pass
    set_busy(True)

    args = ["--model_path", model_dir, "--image_file", file_path, "--desc", desc]
    thread = threading.Thread(target=run_script_in_thread, args=('generate_image_ad.py', '圖像', args), daemon=True)
    thread.start()

def start_realtime_narration():
    """開啟相機，倒數三秒後拍照，並啟動圖像口述影像生成"""
    global _last_selected_image_path

    # 創建一個頂層視窗來顯示相機畫面
    camera_window = tk.Toplevel(app_window)
    camera_window.title("即時影像")
    camera_window.configure(bg="#111827")
    camera_label = tk.Label(camera_window, text="準備開啟相機...", bg="#111827", fg="white", font=("Helvetica", 16))
    camera_label.pack(pady=20, padx=20)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("相機錯誤", "無法開啟相機。", parent=app_window)
        camera_window.destroy()
        return

    def update_frame():
        ret, frame = cap.read()
        if not ret:
            camera_label.config(text="無法讀取影像")
            return
        
        # 顯示影像
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img_tk = ImageTk.PhotoImage(image=img)
        camera_label.config(image=img_tk)
        camera_label.image = img_tk
        if camera_window.winfo_exists():
            camera_window.after(10, update_frame)

    def countdown_and_capture():
        for i in range(3, 0, -1):
            if not camera_window.winfo_exists(): return
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                
                # 在畫面上加入倒數數字
                img_with_text = img.copy()
                from PIL import ImageDraw, ImageFont
                draw = ImageDraw.Draw(img_with_text)
                try:
                    # 嘗試使用一個常見的字體
                    font = ImageFont.truetype("arial.ttf", 100)
                except IOError:
                    font = ImageFont.load_default()
                
                text_bbox = draw.textbbox((0,0), str(i), font=font)
                text_w = text_bbox[2] - text_bbox[0]
                text_h = text_bbox[3] - text_bbox[1]
                text_x = (img.width - text_w) // 2
                text_y = (img.height - text_h) // 2
                draw.text((text_x, text_y), str(i), font=font, fill=(255, 0, 0, 255))
                
                img_tk = ImageTk.PhotoImage(image=img_with_text)
                camera_label.config(image=img_tk)
                camera_label.image = img_tk
                camera_window.update() # 強制更新畫面
            
            time.sleep(1)

        if not camera_window.winfo_exists():
            cap.release()
            return

        ret, frame = cap.read()
        cap.release()
        camera_window.destroy()

        if not ret:
            messagebox.showerror("拍照失敗", "無法擷取影像。", parent=app_window)
            return

        # 儲存照片到暫存檔案
        fd, temp_path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        
        cv2.imwrite(temp_path, frame)
        _last_selected_image_path = temp_path
        
        # 後續流程 (與 start_image_analysis 相似)
        desc = simpledialog.askstring("圖片描述", "請輸入這張圖片的描述或重點：", parent=app_window)
        if desc is None: return
        if not desc.strip():
            messagebox.showwarning("輸入錯誤", "圖片描述不能為空。", parent=app_window)
            return

        model_dir = os.path.join(".", "models", "Llama-3.2-11B-Vision-Instruct")
        if not os.path.isdir(model_dir):
            messagebox.showerror("缺少模型", f"在相對路徑 '{model_dir}' 下找不到 Llama 模型資料夾。\n請確認模型已下載並放置在正確位置。", parent=app_window)
            return
        
        # 清理舊輸出
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
            if realtime_button and realtime_button.winfo_exists(): realtime_button.config(state=tk.DISABLED)
        except tk.TclError: pass
        set_busy(True)

        args = ["--model_path", model_dir, "--image_file", _last_selected_image_path, "--desc", desc]
        thread = threading.Thread(target=run_script_in_thread, args=('generate_image_ad.py', '即時圖像', args), daemon=True)
        thread.start()

    # 延遲一下，讓相機視窗先顯示
    camera_window.after(100, update_frame)
    # 再延遲一下，開始倒數
    threading.Timer(0.5, countdown_and_capture).start()


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

    file_path = filedialog.askopenfilename(
        title="請選擇一個影片",
        filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
    )
    if not file_path: return

    # 禁用按鈕並設定忙碌
    try:
        if image_button and image_button.winfo_exists(): image_button.config(state=tk.DISABLED)
        if video_button and video_button.winfo_exists(): video_button.config(state=tk.DISABLED)
        if realtime_button and realtime_button.winfo_exists(): realtime_button.config(state=tk.DISABLED)
    except tk.TclError: pass
    set_busy(True)

    args = ["--video_path", file_path]
    thread = threading.Thread(target=run_script_in_thread, args=('generate_video_ad.py', '影片', args), daemon=True)
    thread.start()


def main():
    """主函式：建立並執行 GUI"""
    global result_text_widget, status_label_var, app_window, image_button, video_button, realtime_button
    global image_preview_label, narration_output_widget, video_preview_label, progress_bar, status_bar

    # --- 主視窗 ---
    root = tk.Tk()
    app_window = root
    root.title("多媒體口述影像生成工具")
    root.geometry("1024x768")
    root.configure(bg="#111827")
    root.protocol("WM_DELETE_WINDOW", on_closing)

    # --- 風格設定 ---
    style = ttk.Style()
    style.theme_use('clam')
    style.configure("TFrame", background="#111827")
    style.configure("TLabel", background="#111827", foreground="white", font=("Helvetica", 10))
    style.configure("TButton", background="#3b82f6", foreground="white", font=("Helvetica", 10, "bold"), borderwidth=1, focusthickness=3, focuscolor='#93c5fd')
    style.map("TButton", background=[('active', '#2563eb'), ('disabled', '#4b5563')])
    style.configure("TProgressbar", background="#3b82f6", troughcolor="#1f2937")

    # --- 頂部按鈕 ---
    top_button_frame = ttk.Frame(root, style="TFrame")
    top_button_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(10, 5))

    image_button = ttk.Button(top_button_frame, text="生成圖像口述影像", command=start_image_analysis, style="TButton")
    image_button.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)
    ToolTip(image_button, "從本地選擇一張圖片，並為其生成口述影像。")

    video_button = ttk.Button(top_button_frame, text="生成影片口述影像", command=start_video_analysis, style="TButton")
    video_button.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)
    ToolTip(video_button, "從本地選擇一段影片，並為其生成包含口述影像的新影片。")
    
    realtime_button = ttk.Button(top_button_frame, text="生成即時口述影像", command=start_realtime_narration, style="TButton")
    realtime_button.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)
    ToolTip(realtime_button, "開啟相機，倒數三秒後拍照，並為其生成口述影像。")

    # --- 主內容區 (左右分割) ---
    main_pane = tk.PanedWindow(root, orient=tk.HORIZONTAL, bg="#1f2937", sashwidth=8, sashrelief=tk.RAISED)
    main_pane.pack(expand=True, fill=tk.BOTH, padx=10, pady=5)

    # --- 左側：預覽區 ---
    preview_frame = ttk.Frame(main_pane, style="TFrame")
    preview_frame.column_configure(0, weight=1)
    preview_frame.row_configure(1, weight=1)
    main_pane.add(preview_frame, stretch="always")

    preview_label = ttk.Label(preview_frame, text="預覽 / 結果", font=("Helvetica", 14, "bold"))
    preview_label.grid(row=0, column=0, sticky="w", pady=(0, 10))

    # 使用 Notebook (頁籤) 來切換圖片和影片預覽
    preview_notebook = ttk.Notebook(preview_frame)
    preview_notebook.grid(row=1, column=0, sticky="nsew")

    # 圖片頁籤
    image_tab = ttk.Frame(preview_notebook)
    image_tab.column_configure(0, weight=1)
    image_tab.row_configure(0, weight=1) # 讓圖片置中
    image_preview_label = tk.Label(image_tab, bg="#1f2937", text="圖片預覽將顯示於此", fg="gray")
    image_preview_label.grid(row=0, column=0, sticky="nsew")
    preview_notebook.add(image_tab, text="圖像")

    # 影片頁籤
    video_tab = ttk.Frame(preview_notebook)
    video_tab.column_configure(0, weight=1)
    video_tab.row_configure(0, weight=1)
    video_preview_label = tk.Label(video_tab, bg="#1f2937", text="影片預覽將顯示於此", fg="gray")
    video_preview_label.grid(row=0, column=0, sticky="nsew")
    preview_notebook.add(video_tab, text="影片")

    # --- 右側：口述影像文字和日誌 ---
    right_pane = ttk.Frame(main_pane, style="TFrame")
    right_pane.column_configure(0, weight=1)
    right_pane.row_configure(1, weight=3) # 讓日誌區域更大
    right_pane.row_configure(3, weight=2)
    main_pane.add(right_pane, stretch="always")

    narration_label = ttk.Label(right_pane, text="生成的口述影像", font=("Helvetica", 14, "bold"))
    narration_label.grid(row=0, column=0, sticky="w", pady=(0, 5))

    narration_output_widget = scrolledtext.ScrolledText(
        right_pane, wrap=tk.WORD, state=tk.DISABLED, height=8,
        bg="#1f2937", fg="white", font=("Helvetica", 11), relief=tk.SOLID, borderwidth=1,
        highlightbackground="#374151", highlightcolor="#3b82f6", insertbackground="white"
    )
    narration_output_widget.grid(row=1, column=0, sticky="nsew", pady=(0, 10))

    log_label = ttk.Label(right_pane, text="執行日誌", font=("Helvetica", 14, "bold"))
    log_label.grid(row=2, column=0, sticky="w", pady=(5, 5))

    result_text_widget = scrolledtext.ScrolledText(
        right_pane, wrap=tk.WORD, state=tk.DISABLED,
        bg="#1f2937", fg="#d1d5db", font=("Courier New", 9), relief=tk.SOLID, borderwidth=1,
        highlightbackground="#374151", highlightcolor="#3b82f6", insertbackground="white"
    )
    result_text_widget.grid(row=3, column=0, sticky="nsew")

    # --- 底部狀態列和進度條 ---
    status_bar = ttk.Frame(root, style="TFrame", height=25)
    status_bar.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=(5, 5))
    status_label_var = tk.StringVar()
    status_label_var.set("準備就緒")
    status_label = ttk.Label(status_bar, textvariable=status_label_var, anchor=tk.W)
    status_label.pack(side=tk.LEFT)

    progress_bar = ttk.Progressbar(root, mode='indeterminate', style="TProgressbar")
    # progress_bar.pack(side=tk.BOTTOM, fill=tk.X) # 初始隱藏

    if VOICE_ENABLED:
        # 啟動語音指令監聽
        voice_thread = threading.Thread(target=listen_for_commands, daemon=True)
        voice_thread.start()
        speak("歡迎使用多媒體口述影像生成工具。您可以說「圖像」或「影片」來開始。")

    root.mainloop()

def listen_for_commands():
    """在背景監聽語音指令"""
    vc = VoiceCommands()
    vc.add_command("圖像", start_image_analysis)
    vc.add_command("圖片", start_image_analysis)
    vc.add_command("影片", start_video_analysis)
    vc.add_command("即時", start_realtime_narration) # 新增語音指令
    # 可以加入更多指令，例如 vc.add_command("結束", on_closing)

    while True:
        update_status_safe("正在監聽語音指令...")
        text = voice_input("請說出指令...", app_window)
        if text:
            update_status_safe(f"辨識到: '{text}'")
            action = vc.parse(text)
            if callable(action):
                speak(f"收到指令: {text}")
                # 在主執行緒中安全地呼叫 GUI 函式
                if app_window and app_window.winfo_exists():
                    app_window.after(0, action)
            else:
                speak(f"無法辨識指令: {text}")
        time.sleep(0.1)


def on_closing():
    """處理視窗關閉事件"""
    if messagebox.askokcancel("退出", "確定要離開程式嗎？"):
        if app_window:
            app_window.destroy()
    # 清理資源
    stop_video_playback()
    print("應用程式已關閉。")


if __name__ == "__main__":
    main()
