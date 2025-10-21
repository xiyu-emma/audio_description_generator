# voice_interface.py (修正後完整版)

import os
import re
import speech_recognition as sr
import asyncio
import nest_asyncio
import edge_tts
import time
import pygame
import json
from datetime import datetime
import winsound
import traceback # 新增
import uuid

# --------------------------------------------------------------------------
#                           【語音設定】
# --------------------------------------------------------------------------
LANG_CONFIG = {
    "zh-TW": {"voice": "zh-TW-HsiaoChenNeural"},
    "en-US": {"voice": "en-US-JennyNeural"}
}

# --------------------------------------------------------------------------
#                           【全語音UX系統】
# --------------------------------------------------------------------------
class VoiceUXSystem:
    def __init__(self):
        self.speech_rate = 1.0
        self.volume = 1.0
        self.enable_sound_cues = True
        self.beginner_mode = True
        self.load_settings()

    def load_settings(self):
        try:
            if os.path.exists("voice_settings.json"):
                with open("voice_settings.json", 'r', encoding='utf-8') as f:
                    settings = json.load(f)
                    self.speech_rate = settings.get("speech_rate", 1.0)
                    self.volume = settings.get("volume", 1.0)
                    self.beginner_mode = settings.get("beginner_mode", True)
        except Exception as e:
            print(f"讀取設定檔時發生錯誤: {e}")

    def save_settings(self):
        # (在此專案中，設定模組暫不整合，但保留函式)
        pass

voice_ux = VoiceUXSystem()

# --------------------------------------------------------------------------
#                           【音效系統】
# --------------------------------------------------------------------------
class AudioFeedback:
    @staticmethod
    def beep_success():
        if voice_ux.enable_sound_cues:
            try:
                winsound.Beep(800, 150); time.sleep(0.1); winsound.Beep(1200, 150)
            except Exception as e: print(f"播放成功音效失敗: {e}")

    @staticmethod
    def beep_error():
        if voice_ux.enable_sound_cues:
            try: winsound.Beep(300, 300)
            except Exception as e: print(f"播放錯誤音效失敗: {e}")

    @staticmethod
    def beep_listening():
        if voice_ux.enable_sound_cues:
            try: winsound.Beep(600, 100)
            except Exception as e: print(f"播放聆聽音效失敗: {e}")

audio = AudioFeedback()

# --------------------------------------------------------------------------
#                           【初始化】
# --------------------------------------------------------------------------
try:
    nest_asyncio.apply()
except RuntimeError:
    pass # 如果已經應用過，忽略錯誤

try:
    pygame.mixer.init()
    print("Pygame mixer 初始化成功。")
except pygame.error as e:
    print(f"[嚴重警告] Pygame mixer 初始化失敗: {e}")
    print("  -> 語音播放功能將無法使用。請檢查您的音訊設備或驅動程式。")
    # 可以選擇禁用語音功能或退出程式
    # sys.exit(1)

# --------------------------------------------------------------------------
#                           【核心語音功能】
# --------------------------------------------------------------------------
def detect_language(text):
    if not text: return "zh-TW" # 預設中文
    english_chars = len(re.findall(r'[a-zA-Z]', text))
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    return "en-US" if english_chars > chinese_chars else "zh-TW"

def speak(text, wait=True):
    """增強版語音輸出，包含錯誤處理和事件迴圈兼容"""
    if not text or not text.strip(): return
    if not pygame.mixer.get_init():
        print("[錯誤] Pygame mixer 未初始化，無法播放語音。")
        return

    speech_rate = voice_ux.speech_rate
    lang_code = detect_language(text)
    voice = LANG_CONFIG[lang_code]["voice"]

    temp_dir = "temp_audio"
    os.makedirs(temp_dir, exist_ok=True)
    # 使用 UUID 確保檔名唯一性，避免衝突
    output_file = os.path.join(temp_dir, f"speech_{uuid.uuid4()}.mp3")

    async def _generate_speech():
        try:
            rate_str = f"+{int((speech_rate - 1) * 100)}%" if speech_rate >= 1 else f"-{int((1 - speech_rate) * 100)}%"
            communicate = edge_tts.Communicate(text, voice, rate=rate_str)
            await communicate.save(output_file)

            # 播放前再次檢查 mixer 狀態
            if not pygame.mixer.get_init():
                 print("[警告] Pygame mixer 在生成語音後變為未初始化狀態。")
                 return

            # 載入和播放包在 try...except 中
            try:
                pygame.mixer.music.load(output_file)
                pygame.mixer.music.set_volume(voice_ux.volume)
                pygame.mixer.music.play()
            except pygame.error as pg_err:
                 print(f"Pygame 載入或播放錯誤: {pg_err}")
                 audio.beep_error() # 嘗試播放錯誤音效
                 return # 播放失敗直接返回

            if wait:
                while pygame.mixer.get_init() and pygame.mixer.music.get_busy():
                    await asyncio.sleep(0.1) # 在異步函式中使用 asyncio.sleep

        except edge_tts.NoAudioReceived:
            print(f"語音生成錯誤 (Edge TTS): 未收到音訊。文字: '{text[:30]}...'")
            audio.beep_error()
        except Exception as e:
            print(f"語音生成或播放時發生未知錯誤: {e}")
            traceback.print_exc() # 打印詳細錯誤
            audio.beep_error()
        finally:
            # --- 清理檔案 ---
            # 確保播放（如果開始了）已停止
            try:
                if pygame.mixer.get_init() and pygame.mixer.music.get_busy():
                    pygame.mixer.music.stop()
                if pygame.mixer.get_init():
                    pygame.mixer.music.unload() # 釋放文件鎖定
            except Exception: pass # 忽略 unload 可能的錯誤

            await asyncio.sleep(0.2) # 給系統一點時間釋放文件

            if os.path.exists(output_file):
                try:
                    # 嘗試多次刪除，應對 Windows 檔案鎖定
                    for _ in range(3):
                        try:
                            os.remove(output_file)
                            # print(f"已刪除暫存檔: {output_file}") # 調試用
                            break # 成功刪除則跳出
                        except PermissionError:
                            await asyncio.sleep(0.3)
                    else: # 如果嘗試多次仍失敗
                         print(f"[警告] 無法刪除暫存語音檔 (可能仍被占用): {output_file}")
                except Exception as e:
                     print(f"[警告] 刪除暫存語音檔時發生錯誤: {e}")


    # --- 事件迴圈處理 ---
    try:
        # 嘗試獲取正在運行的事件迴圈
        loop = asyncio.get_running_loop()
        # 如果已有迴圈在運行（例如在 Tkinter 主迴圈的執行緒中），創建任務在該迴圈執行
        loop.create_task(_generate_speech())
    except RuntimeError:
        # 如果沒有正在運行的迴圈（例如單獨測試此模組），則啟動一個新的來執行
        try:
            asyncio.run(_generate_speech())
        except Exception as e:
             print(f"執行 asyncio.run 時出錯: {e}")


def voice_input(prompt, timeout=15):
    """統一的語音輸入介面"""
    if voice_ux.beginner_mode:
        speak(prompt + "。聆聽中，請在提示音後說話")
    else:
        speak(prompt)

    audio.beep_listening()
    result = recognize_speech(timeout)

    if result and result not in ["timeout", "error", "unknown"]:
        audio.beep_success()
        return result

    # 提供更具體的失敗原因
    if result == "timeout":
        speak("沒有聽到聲音，請再說一次")
    elif result == "unknown":
        speak("聽不清楚，請大聲一點")
    else: # "error" 或 None
        speak("語音辨識時發生錯誤，請檢查麥克風或網路連線。")
    audio.beep_error()
    return None

def recognize_speech(timeout=15):
    """核心語音辨識"""
    r = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            # print("正在調整環境噪音...") # 調試用
            r.adjust_for_ambient_noise(source, duration=0.5)
            # print("請開始說話...") # 調試用
            # 使用 listen_in_background 可能更適合 GUI，但 listen 較簡單
            audio_data = r.listen(source, timeout=timeout, phrase_time_limit=10)
            # print("正在辨識...") # 調試用
            text = r.recognize_google(audio_data, language="zh-TW")
            print(f"辨識到: {text.strip()}")
            return text.strip()
    except sr.WaitTimeoutError:
        print("語音輸入超時。")
        return "timeout"
    except sr.UnknownValueError:
        print("Google Speech Recognition 無法理解音訊。")
        return "unknown"
    except sr.RequestError as e:
        print(f"無法從 Google Speech Recognition 服務請求結果: {e}")
        return "error"
    except Exception as e:
        print(f"麥克風或語音辨識時發生未知錯誤: {e}")
        traceback.print_exc()
        return "error"

# --------------------------------------------------------------------------
#                           【快捷指令系統】
# --------------------------------------------------------------------------
class VoiceCommands:
    COMMANDS = {
        # 主選單操作指令
        "生成圖像": "image", "圖像": "image", "圖片": "image",
        "生成影片": "video", "影片": "video",
        # 系統指令
        "結束": "exit", "離開": "exit", "掰掰": "exit",
    }

    @classmethod
    def parse(cls, text):
        if not text: return None
        text_lower = text.lower().strip()
        # 優先完全匹配或包含關鍵字
        for key, value in cls.COMMANDS.items():
            # 使用 lower() 比較確保不分大小寫
            if key.lower() in text_lower:
                print(f"指令解析: '{text}' -> '{value}' (匹配 '{key}')")
                return value
        print(f"指令解析: '{text}' -> 未匹配到關鍵字，返回原文")
        return text # 如果沒有匹配，回傳原始文字 (已轉小寫並去除空白)