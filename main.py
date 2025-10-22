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
camera_button = None
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
        if camera_button and camera_button.winfo_exists(): camera_button.config(state=tk.NORMAL)
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
def take_photo_and_generate_ad():
    """開啟攝像頭拍照並啟動後續的口述影像生成流程"""
    global _last_selected_image_path
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("攝像頭錯誤", "無法開啟攝像頭。請檢查攝像頭是否已連接並被其他應用程式佔用。", parent=app_window)
        return

    cv2.namedWindow("拍照 (按 's' 儲存, 'q' 退出)")
    
    img_counter = 0
    img_path = None

    while True:
        ret, frame = cap.read()
        if not ret:
            messagebox.showerror("攝像頭錯誤", "無法讀取攝像頭畫面。", parent=app_window)
            break
        cv2.imshow("拍照 (按 's' 儲存, 'q' 退出)", frame)

        k = cv2.waitKey(1) & 0xFF
        if k == ord('s'):
            img_name = f"opencv_frame_{img_counter}.png"
            img_path = os.path.join(os.path.dirname(__file__), "captured_images", img_name)
            
            # 確保 captured_images 資料夾存在
            os.makedirs(os.path.dirname(img_path), exist_ok=True)
            
            cv2.imwrite(img_path, frame)
            print(f"圖片已儲存至 {img_path}")
            speak(f"圖片已儲存")
            img_counter += 1
            break
        elif k == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

    if img_path:
        desc = simpledialog.askstring("圖片描述", "請輸入這張圖片的描述或重點：", parent=app_window)
        if desc is None: return
        if not desc.strip():
            messagebox.showwarning("輸入錯誤", "圖片描述不能為空。", parent=app_window)
            return

        model_dir = os.path.join(".", "models", "Meta-Llama-3.2-11B-Instruct-Vision")
        if not os.path.isdir(model_dir):
            messagebox.showerror("缺少模型", f"在相對路徑 '{model_dir}' 下找不到 Llama 模型資料夾。\n請確認模型已下載並放置在正確位置。", parent=app_window)
            return

        _last_selected_image_path = img_path

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
            if camera_button and camera_button.winfo_exists(): camera_button.config(state=tk.DISABLED)
        except tk.TclError: pass
        set_busy(True)

        args = ["--model_path", model_dir, "--image_file", img_path, "--desc", desc]
        thread = threading.Thread(target=run_script_in_thread, args=('generate_image_ad.py', '圖像', args), daemon=True)
        thread.start()

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

    model_dir = os.path.join(".", "models", "Meta-Llama-3.2-11B-Instruct-Vision")
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

    file_path = filedialog.askopenfilename(
        title="請選擇一個影片",
        filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv")]
    )
    if not file_path: return

    # 禁用按鈕
    try:
        if image_button and image_button.winfo_exists(): image_button.config(state=tk.DISABLED)
        if video_button and video_button.winfo_exists(): video_button.config(state=tk.DISABLED)
    except tk.TclError: pass
    set_busy(True)

    args = ["--video_path", file_path]
    thread = threading.Thread(target=run_script_in_thread, args=('generate_video_ad.py', '影片', args), daemon=True)
    thread.start()

# --- GUI 建立 ---
def create_main_window():
    """建立並設定主應用程式視窗"""
    global result_text_widget, status_label_var, app_window, image_button, video_button, \
           image_preview_label, narration_output_widget, video_preview_label, progress_bar, status_bar, \
           camera_button

    app_window = tk.Tk()
    app_window.title("智慧口述影像生成工具")
    app_window.geometry("1280x800")
    app_window.configure(bg="#030712") # 深色背景

    # --- 風格設定 ---
    style = ttk.Style()
    style.theme_use('clam')
    style.configure("TFrame", background="#030712")
    style.configure("TLabel", background="#030712", foreground="white", font=("Helvetica", 10))
    style.configure("TButton", background="#1f2937", foreground="white", font=("Helvetica", 10, "bold"), borderwidth=1, focusthickness=3, focuscolor='#4f46e5')
    style.map("TButton", background=[('active', '#374151')])
    style.configure("Accent.TButton", background="#4f46e5", foreground="white", font=("Helvetica", 11, "bold"))
    style.map("Accent.TButton", background=[('active', '#4338ca')])
    style.configure("TProgressbar", background="#4f46e5", troughcolor="#1f2937")

    # --- 主框架 ---
    main_frame = ttk.Frame(app_window, padding="20")
    main_frame.pack(fill=tk.BOTH, expand=True)

    # --- 頂部按鈕區域 ---
    button_frame = ttk.Frame(main_frame)
    button_frame.pack(fill=tk.X, pady=(0, 20))

    image_button = ttk.Button(button_frame, text="從圖片生成口述影像", command=start_image_analysis, style="Accent.TButton")
    image_button.pack(side=tk.LEFT, padx=10, pady=10)
    ToolTip(image_button, "從本地選擇一張圖片，並根據您的描述生成口述影像。")

    camera_button = ttk.Button(button_frame, text="拍照生成口述影像", command=take_photo_and_generate_ad, style="Accent.TButton")
    camera_button.pack(side=tk.LEFT, padx=10, pady=10)
    ToolTip(camera_button, "開啟攝像頭拍照，並根據您的描述生成口述影像。")

    video_button = ttk.Button(button_frame, text="從影片生成口述影像", command=start_video_analysis)
    video_button.pack(side=tk.LEFT, padx=10, pady=10)
    ToolTip(video_button, "從本地選擇一段影片，自動分析內容並生成完整的口述影像。")

    # --- 內容分割區域 ---
    content_paned_window = tk.PanedWindow(main_frame, orient=tk.HORIZONTAL, bg="#111827", sashwidth=8)
    content_paned_window.pack(fill=tk.BOTH, expand=True)

    # --- 左側預覽區域 ---
    preview_frame = ttk.Frame(content_paned_window, width=600)
    content_paned_window.add(preview_frame, stretch="always")

    # 圖片預覽
    image_preview_label = tk.Label(preview_frame, text="圖片預覽區", bg="#111827", fg="grey", height=15)
    image_preview_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 5))
    ttk.Label(preview_frame, text="--- 或 ---").pack()
    # 影片預覽
    video_preview_label = tk.Label(preview_frame, text="影片預覽區", bg="#111827", fg="grey", height=15)
    video_preview_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=(5, 0))

    # --- 右側輸出區域 ---
    output_frame = ttk.Frame(content_paned_window, width=600)
    content_paned_window.add(output_frame, stretch="always")

    # 生成的口述影像文字
    ttk.Label(output_frame, text="生成的口述影像文字:", font=("Helvetica", 12, "bold")).pack(anchor=tk.W, pady=(0, 5))
    narration_output_widget = scrolledtext.ScrolledText(output_frame, wrap=tk.WORD, height=10,
                                                       bg="#1f2937", fg="white", insertbackground="white",
                                                       state=tk.DISABLED, relief=tk.FLAT, borderwidth=2)
    narration_output_widget.pack(fill=tk.BOTH, expand=True)

    # --- 底部日誌區域 ---
    log_frame = ttk.Frame(main_frame)
    log_frame.pack(fill=tk.BOTH, expand=True, pady=(20, 0))
    ttk.Label(log_frame, text="執行日誌:", font=("Helvetica", 12, "bold")).pack(anchor=tk.W, pady=(0, 5))
    result_text_widget = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, height=10,
                                                   bg="#1f2937", fg="#a5b4fc", insertbackground="white",
                                                   state=tk.DISABLED, relief=tk.FLAT, borderwidth=2)
    result_text_widget.pack(fill=tk.BOTH, expand=True)
    
    # --- 狀態列和進度條 ---
    # 進度條 (預設隱藏)
    progress_bar = ttk.Progressbar(main_frame, mode='indeterminate', style="TProgressbar")
    
    # 狀態列
    status_bar = ttk.Frame(main_frame)
    status_bar.pack(side=tk.BOTTOM, fill=tk.X, pady=(5,0))
    status_label_var = tk.StringVar()
    status_label = ttk.Label(status_bar, textvariable=status_label_var)
    status_label.pack(side=tk.LEFT)
    status_label_var.set("就緒")

    return app_window

def on_closing():
    """處理視窗關閉事件"""
    if messagebox.askokcancel("退出", "您確定要退出應用程式嗎?", parent=app_window):
        app_window.destroy()
    
    # 清理資源
    stop_video_playback()
    print("應用程式已關閉。")

if __name__ == "__main__":
    app_window = create_main_window()
    app_window.protocol("WM_DELETE_WINDOW", on_closing)
    try:
        app_window.mainloop()
    except KeyboardInterrupt:
        print("應用程式被使用者中斷。")
    finally:
        on_closing()
