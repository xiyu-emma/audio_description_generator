# enhanced_ui.py - UI增強組件
import tkinter as tk
from tkinter import ttk, Canvas, Frame, messagebox
import cv2
import os

class EnhancedVideoPlayer:
    def __init__(self, parent_frame):
        self.parent_frame = parent_frame
        self.video_cap = None
        self.is_playing = False
        self.current_frame = 0
        self.total_frames = 0
        self.fps = 30.0
        self.video_path = None
        self.after_job = None
        
        # 建立影片播放器UI
        self.setup_ui()
        
    def setup_ui(self):
        # 主要容器
        self.main_frame = ttk.Frame(self.parent_frame)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 影片顯示區域
        self.video_frame = Frame(self.main_frame, bg="#111827", relief=tk.SUNKEN, bd=2)
        self.video_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.video_label = tk.Label(self.video_frame, text="影片播放區域\n點擊播放生成的口述影像影片", 
                                   bg="#111827", fg="grey", font=("Helvetica", 14))
        self.video_label.pack(fill=tk.BOTH, expand=True)
        
        # 控制列
        self.control_frame = ttk.Frame(self.main_frame)
        self.control_frame.pack(fill=tk.X, pady=(5, 0))
        
        # 播放/暫停按鈕
        self.play_button = ttk.Button(self.control_frame, text="▶ 播放", 
                                     command=self.toggle_playback, state=tk.DISABLED)
        self.play_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # 停止按鈕
        self.stop_button = ttk.Button(self.control_frame, text="⏹ 停止", 
                                     command=self.stop_video, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # 進度條
        self.progress_var = tk.DoubleVar()
        self.progress_scale = ttk.Scale(self.control_frame, from_=0, to=100, 
                                       orient=tk.HORIZONTAL, variable=self.progress_var,
                                       command=self.seek_video)
        self.progress_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        
        # 時間標籤
        self.time_label = ttk.Label(self.control_frame, text="00:00 / 00:00")
        self.time_label.pack(side=tk.RIGHT)
        
        # 外部播放器按鈕
        self.external_button = ttk.Button(self.control_frame, text="外部播放", 
                                         command=self.open_external, state=tk.DISABLED)
        self.external_button.pack(side=tk.RIGHT, padx=(0, 10))
    
    def load_video(self, video_path):
        """載入影片"""
        self.stop_video()  # 先停止當前播放
        
        if not os.path.exists(video_path):
            print(f"影片檔案不存在: {video_path}")
            return False
            
        try:
            self.video_cap = cv2.VideoCapture(video_path)
            if not self.video_cap.isOpened():
                print(f"無法開啟影片: {video_path}")
                return False
                
            self.video_path = video_path
            self.total_frames = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.video_cap.get(cv2.CAP_PROP_FPS) or 30.0
            self.current_frame = 0
            
            # 啟用控制按鈕
            self.play_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.NORMAL)
            self.external_button.config(state=tk.NORMAL)
            
            # 顯示第一幀
            self.show_frame(0)
            self.update_time_display()
            
            print(f"影片載入成功: {os.path.basename(video_path)}")
            return True
            
        except Exception as e:
            print(f"載入影片時發生錯誤: {e}")
            return False
    
    def show_frame(self, frame_number):
        """顯示指定幀"""
        if not self.video_cap or not self.video_cap.isOpened():
            return
            
        try:
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = self.video_cap.read()
            
            if ret:
                # 轉換色彩空間
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # 計算顯示尺寸（保持寬高比）
                label_width = self.video_label.winfo_width()
                label_height = self.video_label.winfo_height()
                
                if label_width > 1 and label_height > 1:
                    frame_height, frame_width = frame_rgb.shape[:2]
                    
                    # 計算縮放比例
                    scale_w = label_width / frame_width
                    scale_h = label_height / frame_height
                    scale = min(scale_w, scale_h)
                    
                    new_width = int(frame_width * scale)
                    new_height = int(frame_height * scale)
                    
                    # 調整大小
                    try:
                        from PIL import Image, ImageTk
                        pil_image = Image.fromarray(frame_rgb)
                        pil_image = pil_image.resize((new_width, new_height), Image.LANCZOS)
                        
                        # 轉換為 Tkinter 圖片
                        self.current_tk_image = ImageTk.PhotoImage(pil_image)
                        self.video_label.config(image=self.current_tk_image, text="")
                        self.video_label.image = self.current_tk_image
                    except ImportError:
                        print("[警告] 需要 Pillow 函式庫來顯示影片")
                        return
                    
                self.current_frame = frame_number
                
                # 更新進度條
                if self.total_frames > 0:
                    progress = (frame_number / self.total_frames) * 100
                    self.progress_var.set(progress)
                    
        except Exception as e:
            print(f"顯示影片幀時發生錯誤: {e}")
    
    def toggle_playback(self):
        """切換播放/暫停"""
        if self.is_playing:
            self.pause_video()
        else:
            self.play_video()
    
    def play_video(self):
        """播放影片"""
        if not self.video_cap or not self.video_cap.isOpened():
            return
            
        self.is_playing = True
        self.play_button.config(text="⏸ 暫停")
        self.update_video_playback()
    
    def pause_video(self):
        """暫停影片"""
        self.is_playing = False
        self.play_button.config(text="▶ 播放")
        if self.after_job and hasattr(self, 'app_window') and self.app_window and self.app_window.winfo_exists():
            self.app_window.after_cancel(self.after_job)
            self.after_job = None
    
    def stop_video(self):
        """停止影片"""
        self.pause_video()
        if self.video_cap and self.video_cap.isOpened():
            self.current_frame = 0
            self.show_frame(0)
            self.progress_var.set(0)
            self.update_time_display()
    
    def update_video_playback(self):
        """更新影片播放"""
        if not self.is_playing or not self.video_cap or not self.video_cap.isOpened():
            return
            
        if self.current_frame < self.total_frames - 1:
            self.current_frame += 1
            self.show_frame(self.current_frame)
            self.update_time_display()
            
            # 計算下一幀的延遲
            delay = int(1000 / max(1.0, self.fps))
            if hasattr(self, 'app_window') and self.app_window and self.app_window.winfo_exists():
                self.after_job = self.app_window.after(delay, self.update_video_playback)
        else:
            # 影片播放完畢
            self.pause_video()
    
    def seek_video(self, value):
        """拖拽進度條"""
        if not self.video_cap or not self.video_cap.isOpened():
            return
            
        try:
            progress = float(value)
            frame_number = int((progress / 100) * self.total_frames)
            frame_number = max(0, min(frame_number, self.total_frames - 1))
            
            self.show_frame(frame_number)
            self.update_time_display()
            
        except Exception as e:
            print(f"拖拽進度條時發生錯誤: {e}")
    
    def update_time_display(self):
        """更新時間顯示"""
        if not self.video_cap or self.total_frames == 0:
            return
            
        current_seconds = self.current_frame / self.fps
        total_seconds = self.total_frames / self.fps
        
        current_time = f"{int(current_seconds//60):02d}:{int(current_seconds%60):02d}"
        total_time = f"{int(total_seconds//60):02d}:{int(total_seconds%60):02d}"
        
        self.time_label.config(text=f"{current_time} / {total_time}")
    
    def open_external(self):
        """用外部播放器開啟"""
        if self.video_path and os.path.exists(self.video_path):
            # 呼叫外部函數 open_video_external()
            import subprocess
            import sys
            try:
                if sys.platform.startswith('win'): 
                    os.startfile(self.video_path)
                elif sys.platform == 'darwin': 
                    subprocess.Popen(['open', self.video_path])
                else: 
                    subprocess.Popen(['xdg-open', self.video_path])
            except Exception as e:
                print(f"開啟外部播放器失敗: {e}")
        else:
            messagebox.showwarning("無法開啟", "沒有可開啟的影片檔案。")
    
    def set_app_window(self, app_window):
        """設定主應用程式視窗參考"""
        self.app_window = app_window
    
    def cleanup(self):
        """清理資源"""
        self.pause_video()
        if self.video_cap:
            self.video_cap.release()
            self.video_cap = None
        
        # 重置UI
        self.video_label.config(image='', text="影片播放區域\n點擊播放生成的口述影像影片")
        self.video_label.image = None
        self.play_button.config(state=tk.DISABLED, text="▶ 播放")
        self.stop_button.config(state=tk.DISABLED)
        self.external_button.config(state=tk.DISABLED)
        self.progress_var.set(0)
        self.time_label.config(text="00:00 / 00:00")


class EnhancedImageDisplay:
    def __init__(self, parent_frame):
        self.parent_frame = parent_frame
        self.current_image = None
        self.current_image_tk = None
        self.setup_ui()
    
    def setup_ui(self):
        # 主要容器
        self.main_frame = ttk.Frame(self.parent_frame)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 圖片顯示區域（帶捲軸）
        self.canvas_frame = Frame(self.main_frame, bg="#111827", relief=tk.SUNKEN, bd=2)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        # 建立Canvas和捲軸
        self.canvas = Canvas(self.canvas_frame, bg="#111827", highlightthickness=0)
        self.v_scrollbar = ttk.Scrollbar(self.canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.h_scrollbar = ttk.Scrollbar(self.canvas_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        
        self.canvas.configure(yscrollcommand=self.v_scrollbar.set, xscrollcommand=self.h_scrollbar.set)
        
        # 佈局捲軸和Canvas
        self.v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # 預設提示文字
        self.canvas.create_text(400, 200, text="圖片顯示區域\n選擇圖片後將在此顯示完整圖像", 
                               fill="grey", font=("Helvetica", 14), anchor=tk.CENTER, tags="placeholder")
        
        # 綁定滾輪事件
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind("<Button-4>", self._on_mousewheel)
        self.canvas.bind("<Button-5>", self._on_mousewheel)
    
    def display_image(self, image_path):
        """顯示完整圖像"""
        try:
            from PIL import Image, ImageTk
            
            # 載入圖像
            self.current_image = Image.open(image_path)
            
            # 獲取Canvas尺寸
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            
            if canvas_width <= 1 or canvas_height <= 1:
                # Canvas尺寸未初始化，使用預設值
                canvas_width, canvas_height = 800, 600
            
            # 計算顯示尺寸（可以超出Canvas尺寸以支援捲動）
            img_width, img_height = self.current_image.size
            
            # 設定最大顯示尺寸（比Canvas稍大以允許捲動）
            max_display_width = max(canvas_width, img_width)
            max_display_height = max(canvas_height, img_height)
            
            # 如果圖像太大，適當縮放但保持原始比例
            if img_width > max_display_width * 2 or img_height > max_display_height * 2:
                scale_w = (max_display_width * 1.5) / img_width
                scale_h = (max_display_height * 1.5) / img_height
                scale = min(scale_w, scale_h)
                
                new_width = int(img_width * scale)
                new_height = int(img_height * scale)
                display_image = self.current_image.resize((new_width, new_height), Image.LANCZOS)
            else:
                display_image = self.current_image
            
            # 轉換為Tkinter圖片
            self.current_image_tk = ImageTk.PhotoImage(display_image)
            
            # 清除Canvas並顯示圖片
            self.canvas.delete("all")
            img_id = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.current_image_tk)
            
            # 設定Canvas捲動區域
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))
            
            print(f"圖像顯示成功: {os.path.basename(image_path)}")
            
        except ImportError:
            print("[警告] 需要 Pillow 函式庫來顯示圖片")
            self.canvas.delete("all")
            self.canvas.create_text(400, 200, text="需要 Pillow 函式庫\npip install Pillow", 
                                   fill="red", font=("Helvetica", 12), anchor=tk.CENTER)
        except Exception as e:
            print(f"顯示圖像時發生錯誤: {e}")
            self.canvas.delete("all")
            self.canvas.create_text(400, 200, text=f"圖像載入失敗\n{str(e)}", 
                                   fill="red", font=("Helvetica", 12), anchor=tk.CENTER)
    
    def clear_display(self):
        """清除顯示"""
        self.canvas.delete("all")
        self.canvas.create_text(400, 200, text="圖片顯示區域\n選擇圖片後將在此顯示完整圖像", 
                               fill="grey", font=("Helvetica", 14), anchor=tk.CENTER, tags="placeholder")
        self.current_image = None
        self.current_image_tk = None
    
    def _on_mousewheel(self, event):
        """滑鼠滾輪事件處理"""
        if event.num == 4 or event.delta > 0:
            self.canvas.yview_scroll(-1, "units")
        elif event.num == 5 or event.delta < 0:
            self.canvas.yview_scroll(1, "units")
