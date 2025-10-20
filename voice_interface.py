# voice_interface.py
# 從 audio.py 提煉出的核心語音功能模組

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
        if voice_ux.enable_sound_cues: winsound.Beep(800, 150); time.sleep(0.1); winsound.Beep(1200, 150)
    
    @staticmethod
    def beep_error():
        if voice_ux.enable_sound_cues: winsound.Beep(300, 300)
    
    @staticmethod
    def beep_listening():
        if voice_ux.enable_sound_cues: winsound.Beep(600, 100)

audio = AudioFeedback()

# --------------------------------------------------------------------------
#                           【初始化】
# --------------------------------------------------------------------------
try:
    nest_asyncio.apply()
except RuntimeError:
    pass
pygame.mixer.init()

# --------------------------------------------------------------------------
#                           【核心語音功能】
# --------------------------------------------------------------------------
def detect_language(text):
    english_chars = len(re.findall(r'[a-zA-Z]', text))
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    return "en-US" if english_chars > chinese_chars else "zh-TW"

def speak(text, wait=True):
    if not text or not text.strip(): return
    
    speech_rate = voice_ux.speech_rate
    lang_code = detect_language(text)
    voice = LANG_CONFIG[lang_code]["voice"]
    
    temp_dir = "temp_audio"
    os.makedirs(temp_dir, exist_ok=True)
    output_file = os.path.join(temp_dir, f"speech_{int(time.time() * 1000)}.mp3")
    
    async def _generate_speech():
        try:
            rate_str = f"+{int((speech_rate - 1) * 100)}%" if speech_rate >= 1 else f"-{int((1 - speech_rate) * 100)}%"
            communicate = edge_tts.Communicate(text, voice, rate=rate_str)
            await communicate.save(output_file)
            
            pygame.mixer.music.load(output_file)
            pygame.mixer.music.set_volume(voice_ux.volume)
            pygame.mixer.music.play()
            
            if wait:
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)
        except Exception as e:
            print(f"語音生成錯誤: {e}")
            audio.beep_error()
        finally:
            if wait:
                # 確保播放完畢後再刪除檔案
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)
                pygame.mixer.music.unload()
                time.sleep(0.1)
                try:
                    if os.path.exists(output_file): os.remove(output_file)
                except: pass
    
    loop = asyncio.get_event_loop()
    if loop.is_running():
        loop.create_task(_generate_speech())
    else:
        loop.run_until_complete(_generate_speech())

def voice_input(prompt, timeout=15):
    if voice_ux.beginner_mode:
        speak(prompt + "。聆聽中，請在提示音後說話")
    else:
        speak(prompt)
    
    audio.beep_listening()
    result = recognize_speech(timeout)
    
    if result and result not in ["timeout", "error", "unknown"]:
        audio.beep_success()
        return result
    
    if result == "timeout":
        speak("沒有聽到聲音，請再說一次")
    elif result == "unknown":
        speak("聽不清楚，請大聲一點")
    else:
        speak("辨識錯誤，請再試一次")
    audio.beep_error()
    return None

def recognize_speech(timeout=15):
    r = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            r.adjust_for_ambient_noise(source, duration=0.5)
            audio_data = r.listen(source, timeout=timeout, phrase_time_limit=10)
            text = r.recognize_google(audio_data, language="zh-TW")
            print(f"辨識到: {text.strip()}") # 除錯用
            return text.strip()
    except sr.WaitTimeoutError: return "timeout"
    except sr.UnknownValueError: return "unknown"
    except Exception: return "error"

# --------------------------------------------------------------------------
#                           【快捷指令系統】
# --------------------------------------------------------------------------
class VoiceCommands:
    COMMANDS = {
        # 新增主選單的操作指令
        "生成圖像": "image",
        "圖像": "image",
        "圖片": "image",
        "生成影片": "video",
        "影片": "video",
        
        # 保留系統指令
        "結束": "exit",
        "離開": "exit",
        "掰掰": "exit",
    }
    
    @classmethod
    def parse(cls, text):
        if not text: return None
        text = text.lower().strip()
        # 尋找關鍵字
        for key, value in cls.COMMANDS.items():
            if key in text:
                return value
        return text # 如果沒有匹配，回傳原始文字