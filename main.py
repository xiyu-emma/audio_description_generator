# main.py (單一環境版本)

import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, simpledialog # 新增
import subprocess
import sys
import os
import threading
import time
import queue # 新增
import traceback # 新增

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
image_folder_var = None
text_folder_var = None
question_var = None
result_text_widget = None
status_label_var = None
app_window = None
image_button = None # 使按鈕成為全域，以便禁用/啟用
video_button = None

# --- GUI 輔助函式 ---
def select_folder(target_var, title="選擇資料夾"):
    folder_selected = filedialog.askdirectory(title=title)
    if folder_selected:
        target_var.set(folder_selected)

def update_gui_safe(widget, text):
    if widget and app_window:
        try:
            widget.config(state=tk.NORMAL)
            widget.insert(tk.END, text + "\n")
            widget.see(tk.END)
            widget.config(state=tk.DISABLED)
        except tk.TclError:
            pass # 視窗可能已關閉

def update_status_safe(text):
    if status_label_var and app_window:
        try:
            status_label_var.set(text)
        except tk.TclError:
            pass

# --- 執行緒函式 ---
def run_script_in_thread(script_name: str, script_type: str, args: list):
    """在背景執行緒中執行腳本並將輸出傳回 GUI"""
    update_status_safe(f"正在執行 {script_type} 程序...")
    update_gui_safe(result_text_widget, f"\n--- 開始執行 {script_name} ---")
    speak(f"正在啟動，{script_type}口述影像生成程序")

    try:
        script_path = os.path.join(os.path.dirname(__file__), script_name)
        # 【重要】使用 sys.executable，因為都在同一個環境
        command = [sys.executable, script_path] + args

        print(f"執行指令: {' '.join(command)}")

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding='utf-8', errors='replace',
            bufsize=1, creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
        )

        final_answer = f"[{script_type} 未返回明確答案]"

        # 即時讀取 stdout
        if process.stdout:
            for line in iter(process.stdout.readline, ''):
                print(line, end='')
                if app_window:
                    app_window.after(0, update_gui_safe, result_text_widget, line.strip())
                if line.strip().startswith("FINAL_ANSWER:"):
                    final_answer = line.strip().replace("FINAL_ANSWER:", "").strip()
            process.stdout.close()

        # 讀取 stderr
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
            if app_window and script_name == 'generate_image_ad.py':
                 app_window.after(100, messagebox.showinfo, "圖像分析結果", final_answer)

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
        # 無論成功或失敗，都嘗試重新啟用按鈕
        if app_window:
            app_window.after(100, enable_buttons)

def enable_buttons():
    """重新啟用按鈕"""
    try:
        if image_button: image_button.config(state=tk.NORMAL)
        if video_button: video_button.config(state=tk.NORMAL)
    except tk.TclError:
        pass # 視窗可能已關閉

def start_image_analysis():
    """從 GUI 取得參數並啟動圖像分析執行緒"""
    image_path = image_folder_var.get()
    text_path = text_folder_var.get()
    question = question_var.get()
    model_dir = os.path.join(".", "models", "Llama-3.2-11B-Vision-Instruct") # 使用相對路徑

    if not image_path or not os.path.isdir(image_path):
        messagebox.showwarning("輸入錯誤", "請選擇有效的圖片資料夾。")
        return
    if not text_path or not os.path.isdir(text_path):
        messagebox.showwarning("輸入錯誤", "請選擇有效的文字資料夾。")
        return
    if not question:
        messagebox.showwarning("輸入錯誤", "請輸入您想詢問的問題。")
        return
    if not os.path.isdir(model_dir):
         messagebox.showerror("缺少模型", f"在相對路徑 '{model_dir}' 下找不到 Llama 模型資料夾。\n請確認模型已下載並放置在正確位置。")
         return

    args = [
        "--model_path", model_dir,
        "--image_path", image_path,
        "--text_path", text_path,
        "--question", question
    ]

    result_text_widget.config(state=tk.NORMAL)
    result_text_widget.delete('1.0', tk.END)
    result_text_widget.config(state=tk.DISABLED)

    image_button.config(state=tk.DISABLED)
    video_button.config(state=tk.DISABLED)

    # 啟動背景執行緒
    thread = threading.Thread(
        target=run_script_in_thread,
        args=('generate_image_ad.py', '圖像', args),
        daemon=True
    )
    thread.start()

def start_video_analysis():
    """啟動影片分析執行緒"""
    result_text_widget.config(state=tk.NORMAL)
    result_text_widget.delete('1.0', tk.END)
    result_text_widget.config(state=tk.DISABLED)

    image_button.config(state=tk.DISABLED)
    video_button.config(state=tk.DISABLED)

    # 不需要額外參數
    args = []

    # 啟動背景執行緒
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

    time.sleep(2) # 等待 GUI 完全啟動
    speak("歡迎使用口述影像生成系統")

    session_active = True
    while session_active:
        prompt = "請說出指令：生成圖像、生成影片，或 結束"
        command = voice_input(prompt)

        if command:
            parsed = VoiceCommands.parse(command)

            if parsed == "image":
                # 語音觸發圖像分析需要處理輸入
                speak("好的，請告訴我圖片資料夾、文字資料夾和問題。")
                # 這裡需要更複雜的語音互動來收集路徑和問題，
                # 目前先簡化為提示使用者手動操作
                speak("請先在介面上選擇資料夾並輸入問題，然後說「開始圖像分析」。")
                # 等待確認指令
                confirm_command = voice_input("準備好後請說「開始圖像分析」")
                if confirm_command and "開始圖像分析" in confirm_command:
                     if app_window:
                         # 使用 after 在主執行緒中安全地觸發按鈕點擊
                         app_window.after(0, start_image_analysis)
                     else:
                         speak("無法觸發分析，主視窗似乎已關閉。")
                else:
                    speak("取消圖像分析。")

            elif parsed == "video":
                 if app_window:
                     app_window.after(0, start_video_analysis)
                 else:
                     speak("無法觸發分析，主視窗似乎已關閉。")

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
    global image_folder_var, text_folder_var, question_var, result_text_widget, status_label_var, app_window
    global image_button, video_button

    root = tk.Tk()
    app_window = root
    root.title("口述影像生成系統")
    root.geometry("750x650") # 再稍微加大

    style = ttk.Style()
    style.configure("TButton", font=("Helvetica", 12), padding=5)
    style.configure("TLabel", font=("Helvetica", 10))
    style.configure("Header.TLabel", font=("Helvetica", 14, "bold"))
    style.configure("Folder.TButton", font=("Helvetica", 10))

    main_frame = ttk.Frame(root, padding="15")
    main_frame.pack(expand=True, fill="both")

    image_frame = ttk.LabelFrame(main_frame, text="圖像口述影像 (使用 Llama)", padding="10")
    image_frame.pack(fill="x", pady=(0, 10))

    img_folder_frame = ttk.Frame(image_frame)
    img_folder_frame.pack(fill="x", pady=5)
    ttk.Label(img_folder_frame, text="圖片資料夾:").pack(side="left", padx=(0, 5))
    image_folder_var = tk.StringVar()
    img_folder_entry = ttk.Entry(img_folder_frame, textvariable=image_folder_var, width=60)
    img_folder_entry.pack(side="left", fill="x", expand=True, padx=5)
    img_select_button = ttk.Button(img_folder_frame, text="選擇...", command=lambda: select_folder(image_folder_var, "選擇圖片資料夾"), style="Folder.TButton")
    img_select_button.pack(side="left")

    txt_folder_frame = ttk.Frame(image_frame)
    txt_folder_frame.pack(fill="x", pady=5)
    ttk.Label(txt_folder_frame, text="文字資料夾:").pack(side="left", padx=(0, 5))
    text_folder_var = tk.StringVar()
    txt_folder_entry = ttk.Entry(txt_folder_frame, textvariable=text_folder_var, width=60)
    txt_folder_entry.pack(side="left", fill="x", expand=True, padx=5)
    txt_select_button = ttk.Button(txt_folder_frame, text="選擇...", command=lambda: select_folder(text_folder_var, "選擇文字資料夾"), style="Folder.TButton")
    txt_select_button.pack(side="left")

    question_frame = ttk.Frame(image_frame)
    question_frame.pack(fill="x", pady=5)
    ttk.Label(question_frame, text="詢問的問題:").pack(side="left", padx=(0, 5))
    question_var = tk.StringVar(value="詳細描述這張圖片的內容。")
    question_entry = ttk.Entry(question_frame, textvariable=question_var, width=70)
    question_entry.pack(side="left", fill="x", expand=True, padx=5)

    image_button = ttk.Button(image_frame, text="開始圖像分析 (Llama)", command=start_image_analysis)
    image_button.pack(pady=10)

    video_button = ttk.Button(main_frame, text="開始影片分析 (Gemini)", command=start_video_analysis)
    video_button.pack(fill="x", ipady=8, pady=10)

    result_frame = ttk.LabelFrame(main_frame, text="執行結果與輸出", padding="10")
    result_frame.pack(expand=True, fill="both", pady=(10, 0))
    result_text_widget = scrolledtext.ScrolledText(result_frame, wrap=tk.WORD, height=15, state=tk.DISABLED, font=("Consolas", 9))
    result_text_widget.pack(expand=True, fill="both")

    status_label_var = tk.StringVar(value="準備就緒")
    status_bar = ttk.Label(root, textvariable=status_label_var, relief=tk.SUNKEN, anchor=tk.W, padding=(5, 2))
    status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    return root

# --- 程式主進入點 ---
if __name__ == "__main__":
    app_window = create_gui()

    # 只有在 voice_interface 成功導入時才啟動語音執行緒
    if VOICE_ENABLED:
        voice_thread = threading.Thread(target=voice_interaction_loop, daemon=True)
        voice_thread.start()
    else:
        update_status_safe("語音功能未啟用")

    app_window.mainloop()

    print("應用程式已關閉。")