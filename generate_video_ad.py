# generate_video_ad.py (最終完整修正版 v2)

# --- 匯入函式庫 ---
import os
import cv2
import numpy as np
import time
import random
import shutil
import asyncio
import nest_asyncio
import re
from datetime import timedelta
import sys
import json
import tempfile
import math
from typing import List, Tuple
from tkinter import filedialog, messagebox, simpledialog
import tkinter as tk
import traceback

# --- 核心套件載入 ---
try:
    import scenedetect
    from scenedetect.detectors import ContentDetector
    import google.generativeai as genai
    from PIL import Image
    from google.api_core.exceptions import ResourceExhausted, GoogleAPICallError, NotFound
    import edge_tts
    from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip, vfx
    import moviepy.audio.fx.all as afx # 【核心修正】導入音訊效果模組
    from mutagen.mp3 import MP3
    import whisper
except ImportError as e:
    root = tk.Tk()
    root.withdraw()
    messagebox.showerror(
        "套件缺失錯誤",
        f"錯誤：缺少必要的套件。請先執行 'pip install opencv-python \"scenedetect[opencv]\" numpy google-generativeai Pillow edge-tts moviepy mutagen openai-whisper tk' 來安裝。\n\n詳細錯誤: {e}"
    )
    sys.exit()

nest_asyncio.apply()

# --------------------------------------------------------------------------
#                           【使用者設定區】
# --------------------------------------------------------------------------
try:
    with open('api_key.txt', 'r', encoding='utf-8') as f:
        API_KEY = f.read().strip()
    if not API_KEY:
        raise FileNotFoundError
except FileNotFoundError:
    messagebox.showerror("API 金鑰錯誤", "找不到 'api_key.txt' 檔案，或檔案內容為空。請建立此檔案並填入您的 Google API 金鑰。")
    sys.exit()

SECONDS_PER_CHAR = 0.2682
VOICE = "zh-TW-HsiaoChenNeural"
BACKGROUND_VOLUME = 0.7
NARRATION_VOLUME = 1.9
MAX_SPEEDUP_FACTOR = 1.15
AUTO_CLEANUP_TEMP_FILES = True

# --------------------------------------------------------------------------

def step0_detect_voice_segments(video_path: str, model_size: str = "small", verbose: bool = False) -> List[Tuple[float, float]]:
    print("\n" + "="*50)
    print("--- 步驟 0: 偵測影片中有人聲的區段（使用 Whisper） ---")
    print("="*50)
    if not os.path.exists(video_path):
        print(f"[錯誤] 找不到影片檔案：{video_path}")
        return []
    
    tmp_dir = tempfile.mkdtemp(prefix="video_audio_")
    tmp_audio_path = os.path.join(tmp_dir, "extracted_audio.wav")

    try:
        with VideoFileClip(video_path) as video:
            if video.audio is None:
                print("  - 影片沒有音軌，跳過語音偵測。")
                return []
            print("  - 擷取影片音軌至暫存檔...")
            video.audio.write_audiofile(tmp_audio_path, fps=16000, logger=None)

        print(f"  - 下載 / 載入 Whisper 模型 ({model_size})...")
        model = whisper.load_model(model_size)

        print("  - 呼叫 whisper.transcribe 進行語音辨識與時間戳偵測...")
        result = model.transcribe(tmp_audio_path, word_timestamps=False, verbose=verbose)
        
        segments = result.get("segments", [])
        speech_segments = [(float(s.get("start", 0.0)), float(s.get("end", 0.0))) for s in segments if float(s.get("end", 0.0)) - float(s.get("start", 0.0)) >= 0.05]

        merged = []
        if speech_segments:
            sorted_segments = sorted(speech_segments)
            current_start, current_end = sorted_segments[0]
            for next_start, next_end in sorted_segments[1:]:
                if next_start <= current_end + 0.15:
                    current_end = max(current_end, next_end)
                else:
                    merged.append((current_start, current_end))
                    current_start, current_end = next_start, next_end
            merged.append((current_start, current_end))
        
        speech_segments = merged
        print(f"  - Whisper 偵測到 {len(speech_segments)} 段有人聲區間。")
        for i, (s, e) in enumerate(speech_segments, start=1):
            print(f"    {i}. {s:.3f}s 〜 {e:.3f}s (長度 {(e-s):.3f}s)")
        return speech_segments

    except Exception as e:
        print(f"[警告] 使用 Whisper 偵測語音時發生錯誤: {e}")
        print("  - 這通常代表 ffmpeg 未正確安裝或設定在系統 PATH 中。")
        return []
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

def get_non_dialogue_segments(speech_segments: List[Tuple[float, float]], video_duration: float, min_silence_length: float = 0.25) -> List[Tuple[float, float]]:
    nonspeech_segments = []
    cursor = 0.0
    for s, e in sorted(speech_segments):
        if s - cursor >= min_silence_length:
            nonspeech_segments.append((cursor, s))
        cursor = max(cursor, e)
    if video_duration - cursor >= min_silence_length:
        nonspeech_segments.append((cursor, video_duration))
    return nonspeech_segments

def imwrite_unicode(path, image):
    try:
        is_success, buffer = cv2.imencode(".jpg", image)
        if not is_success: return False
        with open(path, 'wb') as f:
            f.write(buffer)
        return True
    except Exception as e:
        print(f"寫入檔案時發生錯誤: {e}")
        return False

def calculate_sharpness(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def handle_api_call(model, prompt_parts, max_retries=3):
    for retries in range(max_retries):
        try:
            response = model.generate_content(prompt_parts)
            return response
        except (ResourceExhausted, GoogleAPICallError) as e:
            if isinstance(e, ResourceExhausted) or (hasattr(e, 'code') and e.code == 429):
                wait_time = (2 ** (retries + 1)) + random.uniform(0, 1)
                print(f"    [警告] 觸發 API 速率限制。將在 {wait_time:.2f} 秒後重試...")
                time.sleep(wait_time)
            else:
                raise e
    print(f"    [錯誤] API 速率限制達到最大重試次數，放棄。")
    return None

def step1_extract_keyframes(video_path: str, output_dir: str, threshold: float = 27.0):
    print("\n" + "="*50)
    print("--- 步驟 1: 分析影片並擷取關鍵影格 ---")
    print("="*50)
    os.makedirs(output_dir, exist_ok=True)
    try:
        video = scenedetect.open_video(video_path)
        scene_manager = scenedetect.SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=threshold))
        scene_manager.detect_scenes(video=video)
        scene_list = scene_manager.get_scene_list()
        fps = video.frame_rate
        if not scene_list:
            print("[警告] 在影片中偵測不到場景變化，將以固定間隔取樣。")
            duration_sec = video.duration.get_seconds()
            scene_list = [(video.base_timecode + (i*fps*5), video.base_timecode + ((i+1)*fps*5)) for i in range(int(duration_sec/5))]
        
        print(f"分析 {len(scene_list)} 個區段以擷取最佳影格...")
        for _, (start_tc, end_tc) in enumerate(scene_list):
            start_frame, end_frame = start_tc.get_frames(), end_tc.get_frames()
            if start_frame >= end_frame: continue
            
            best_frame, max_sharpness = None, -1
            video.seek(start_frame)
            for _ in range(end_frame - start_frame):
                frame = video.read()
                if frame is None: break
                sharpness = calculate_sharpness(frame)
                if sharpness > max_sharpness:
                    max_sharpness, best_frame = sharpness, frame.copy()
            if best_frame is not None:
                middle_frame_num = start_frame + (end_frame - start_frame) // 2
                total_millis = int((middle_frame_num / fps) * 1000)
                minutes, seconds, millis = total_millis // 60000, (total_millis % 60000) // 1000, total_millis % 1000
                filename = f"{minutes:02d}-{seconds:02d}-{millis:03d}.jpg"
                imwrite_unicode(os.path.join(output_dir, filename), best_frame)
    except Exception as e:
        print(f"[嚴重錯誤] 擷取關鍵影格時發生錯誤: {e}")
        return False
    print("\n[成功] 步驟 1 完成！")
    return True

def step2_generate_initial_descriptions(api_key, image_dir, video_summary, video_duration, nonspeech_segments):
    print("\n" + "="*50)
    print("--- 步驟 2: AI 生成初步描述 ---")
    print("="*50)
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    try:
        image_files_all = sorted([f for f in os.listdir(image_dir) if f.lower().endswith('.jpg')])
    except FileNotFoundError:
        print(f"[錯誤] 找不到關鍵影格目錄：{image_dir}")
        return None
        
    if not image_files_all:
        print("[警告] 影格目錄中沒有圖片")
        return None

    start_times, image_files = [], []
    for filename in image_files_all:
        try:
            base_name = os.path.splitext(filename)[0]
            parts = base_name.split('-')
            if len(parts) == 3:
                start_time = int(parts[0]) * 60 + int(parts[1]) + int(parts[2]) / 1000.0
                start_times.append(start_time)
                image_files.append(filename)
            else:
                print(f"  [警告] 發現不合格式的檔名，已跳過：{filename}")
        except (ValueError, IndexError):
            print(f"  [警告] 解析檔名 {filename} 失敗，已跳過。")

    if not image_files:
        print("[錯誤] 所有關鍵影格的檔名格式都不正確，無法繼續。")
        return None

    descriptions_data, previous_description = [], "這是影片的開頭。"

    def compute_max_chars(start_time, end_time):
        for ns_start, ns_end in nonspeech_segments:
            if ns_start <= start_time < ns_end:
                available_duration = min(ns_end, end_time) - start_time
                return max(8, int(available_duration / SECONDS_PER_CHAR))
        return 8

    for i, filename in enumerate(image_files):
        ideal_start_time = start_times[i]
        end_boundary = start_times[i + 1] if i + 1 < len(start_times) else video_duration
        max_chars = compute_max_chars(ideal_start_time, end_boundary)
        print(f"  處理中 ({i+1}/{len(image_files)}): {filename} (可用 {end_boundary - ideal_start_time:.2f}s, 上限 {max_chars} 字)")
        try:
            image_path = os.path.join(image_dir, filename)
            with Image.open(image_path) as img:
                prompt_parts = [
                    f"你是一位專業的口述影像撰寫者，任務是為影片的關鍵畫面生成連貫的描述。\n\n"
                    f"【影片整體摘要】:\n{video_summary.strip()}\n\n"
                    f"【前一個畫面的描述】:\n{previous_description}\n\n"
                    f"【你的任務】:\n請依據上下文，專注描述下方圖片的內容。生成一句客觀、具體的口述影像。\n\n"
                    f"【絕對規則】\n這句描述的中文總字數【絕對不能超過 {max_chars} 個字】。請嚴格遵守此限制。",
                    img
                ]
                response = handle_api_call(model, prompt_parts)
            
            if not response: continue
            current_description = response.text.strip().replace('\n', ' ')
            previous_description = current_description
            descriptions_data.append({
                "ideal_start_time": ideal_start_time, "text": current_description,
                "available_duration": end_boundary - ideal_start_time, "max_chars": max_chars
            })
            print(f"    -> 生成描述: {current_description}")
            time.sleep(2)
        except NotFound as e:
             print(f"    [嚴重錯誤] Google API 錯誤: {e}")
             print(f"    -> 請確認您的 API 金鑰是否正確，以及 'gemini-2.5-flash' 模型是否可用。")
             return None
        except Exception as e:
            print(f"    [嚴重錯誤] 處理 {filename} 時發生錯誤: {e}")
            traceback.print_exc()
            continue
            
    print(f"\n[成功] 步驟 2 完成！已生成 {len(descriptions_data)} 條初步描述。")
    return descriptions_data

def step3_refine_and_merge_descriptions(api_key, initial_data, video_summary, nonspeech_segments):
    print("\n" + "="*50)
    print("--- 步驟 3: AI 精煉與智慧合併描述 ---")
    print("="*50)
    if not initial_data: return None

    raw_text_for_prompt = ""
    for item in initial_data:
        td = timedelta(seconds=item['ideal_start_time'])
        minutes, remainder = divmod(td.seconds, 60)
        ms = td.microseconds // 1000
        end_time = item['ideal_start_time'] + item['available_duration']
        
        max_chars = 8
        for ns_start, ns_end in nonspeech_segments:
            if ns_start <= item['ideal_start_time'] < ns_end:
                available_duration = min(ns_end, end_time) - item['ideal_start_time']
                max_chars = max(8, int(available_duration / SECONDS_PER_CHAR))
                break
        
        raw_text_for_prompt += f"{minutes:02d}:{remainder:02d}:{ms:03d}: {item['text']} (字數上限: {max_chars} 字)\n"

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash') 
        prompt = f"""
你是一位頂尖的口述影像編輯師，請參考【影片整體摘要】，將下方的【原始描述】進行專業級的優化。
【核心任務】
1. 智慧合併：若多個描述時間很接近且描述同一連續動作，請合併成一句，並用最早的時間。
2. 流暢潤飾：讓句子更自然、更具描述性。
3. 嚴格字數限制：每一條描述【必須遵守原始描述括號中提供的字數上限】。
【格式要求】
* 每行輸出嚴格保持 `mm:ss:ms: 描述內容` 格式。
* 直接輸出結果，不要任何引言或結語。
【影片整體摘要】:
{video_summary.strip()}
【原始描述】:
{raw_text_for_prompt}
"""
        print("正在呼叫 AI 進行智慧合併與精煉...")
        response = handle_api_call(model, prompt)
        if not response: raise Exception("AI 精煉步驟的 API 呼叫失敗。")
        
        refined_text_from_ai = response.text.strip()
        refined_descriptions = []
        pattern = re.compile(r"^\s*(\d{2}):(\d{2}):(\d{3})\s*[:\-]?\s*(.+)")
        for line in refined_text_from_ai.split('\n'):
            match = pattern.match(line.strip())
            if match:
                m, s, ms, desc = match.groups()
                start_time = int(m) * 60 + int(s) + int(ms) / 1000.0
                refined_descriptions.append({"ideal_start_time": start_time, "text": desc.strip()})
        
        if not refined_descriptions and refined_text_from_ai:
            print("[警告] AI 回應的格式不符合預期，將使用步驟2的原始描述。")
            return initial_data

        print(f"AI 精煉完成！描述數量從 {len(initial_data)} 條優化為 {len(refined_descriptions)} 條。")
        print("\n[成功] 步驟 3 完成！")
        return refined_descriptions

    except Exception as e:
        print(f"[嚴重錯誤] 在 AI 精煉過程中發生錯誤: {e}")
        traceback.print_exc()
        print("[警告] 步驟3失敗，將使用步驟2的原始描述繼續。")
        return initial_data

async def _run_tts_tasks(descriptions):
    semaphore = asyncio.Semaphore(5) 
    async def safe_tts_task(desc, index):
        async with semaphore:
            try:
                print(f"  - (TTS 任務 {index}) 開始生成: '{desc['text'][:20]}...'")
                communicate = edge_tts.Communicate(desc['text'], VOICE)
                await communicate.save(desc['audio_path'])
            except Exception as e:
                print(f"  - [警告] (TTS 任務 {index}) 生成失敗: '{desc['text'][:20]}...'。錯誤: {e}")

    tasks = [safe_tts_task(desc, i + 1) for i, desc in enumerate(descriptions)]
    await asyncio.gather(*tasks)

def step4_generate_audio_and_measure_duration(descriptions):
    print("\n" + "="*50)
    print("--- 步驟 4: 生成語音並測量時長 ---")
    print("="*50)
    if not descriptions: return None

    global TEMP_AUDIO_DIR
    os.makedirs(TEMP_AUDIO_DIR, exist_ok=True)
    for i, desc in enumerate(descriptions):
        desc['audio_path'] = os.path.join(TEMP_AUDIO_DIR, f"narration_{i}.mp3")
    
    print("開始並行生成所有語音檔...")
    try:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(_run_tts_tasks(descriptions))
    except Exception as e:
        print(f"  [嚴重錯誤] 非同步任務執行時發生未知錯誤: {e}")

    print("所有語音檔生成完畢！\n正在測量每個語音檔的實際長度...")
    successful_descriptions = []
    for desc in descriptions:
        try:
            if os.path.exists(desc['audio_path']):
                audio = MP3(desc['audio_path'])
                desc['audio_duration'] = audio.info.length
                print(f"  - {os.path.basename(desc['audio_path'])}: {desc['audio_duration']:.2f} 秒")
                successful_descriptions.append(desc)
            else:
                print(f"  - [跳過] {os.path.basename(desc['audio_path'])} 因生成失敗而無法測量。")
        except Exception as e:
            desc['audio_duration'] = 0.0
            print(f"  - [錯誤] 測量 {os.path.basename(desc['audio_path'])} 時長失敗: {e}")
            
    print(f"\n[成功] 步驟 4 完成！成功生成並測量了 {len(successful_descriptions)} 個語音檔。")
    return successful_descriptions

def step5_plan_timeline(descriptions, video_duration, non_dialogue_segments):
    print("\n" + "="*50)
    print("--- 步驟 5: 動態規劃旁白時間軸 ---")
    print(f"影片總長度: {video_duration:.2f} 秒，最大加速容忍度: {MAX_SPEEDUP_FACTOR}")
    print("="*50)
    if not descriptions: return None
    if not non_dialogue_segments:
        non_dialogue_segments = [(0, video_duration)]
    
    seg_cursors = {i: seg[0] for i, seg in enumerate(non_dialogue_segments)}
    planned_descriptions = []

    for i, desc in enumerate(sorted(descriptions, key=lambda x: x['ideal_start_time'])):
        desired_start = desc.get('ideal_start_time', 0.0)
        audio_duration = desc.get('audio_duration', 0.0)
        placed = False
        
        # 尋找所有未來可用的區段
        available_segments = [
            (idx, seg_start, seg_end) for idx, (seg_start, seg_end) in enumerate(non_dialogue_segments)
            if seg_end > desired_start
        ]

        # 按照與 ideal_start_time 的接近程度排序
        available_segments.sort(key=lambda x: abs(x[1] - desired_start))

        for seg_idx, seg_start, seg_end in available_segments:
            cursor = max(seg_cursors.get(seg_idx, seg_start), desired_start)
            if cursor >= seg_end: continue

            available_duration = seg_end - cursor
            
            if available_duration >= audio_duration:
                desc.update({'final_start_time': cursor, 'speed_factor': 1.0, 'final_clip_duration': audio_duration})
                seg_cursors[seg_idx] = cursor + audio_duration
                print(f"  - 第 {i+1} 句: (時間 {cursor:.2f}s) 正常播放。")
                placed = True
                break
            elif available_duration > 0.5 and (audio_duration / available_duration) <= MAX_SPEEDUP_FACTOR:
                required_speedup = audio_duration / available_duration
                desc.update({'final_start_time': cursor, 'speed_factor': required_speedup, 'final_clip_duration': available_duration})
                seg_cursors[seg_idx] = cursor + available_duration
                print(f"  - [注意] 第 {i+1} 句: (時間 {cursor:.2f}s) 空間不足，加速 {required_speedup:.2f} 倍。")
                placed = True
                break

        if placed:
            planned_descriptions.append(desc)
        else:
            print(f"  - [警告] 第 {i+1} 句: (理想時間 {desired_start:.2f}s) 找不到合適的無人聲區段，捨棄。")

    print(f"\n[成功] 步驟 5 完成！成功規劃了 {len(planned_descriptions)} 條旁白。")
    return sorted(planned_descriptions, key=lambda x: x['final_start_time'])

def step6_synthesize_final_video(video_path, descriptions, output_path):
    print("\n" + "="*50)
    print("--- 步驟 6: 最終影片合成 ---")
    print("="*50)
    
    if not descriptions:
        print("[警告] 沒有旁白可合成，將只複製原影片。")
        shutil.copy(video_path, output_path)
        return True
    
    video_clip = None
    try:
        video_clip = VideoFileClip(video_path)
        narration_clips = []

        for desc in descriptions:
            if 'audio_path' in desc and os.path.exists(desc['audio_path']):
                audio_clip = AudioFileClip(desc['audio_path'])
                speed_factor = desc.get('speed_factor', 1.0)

                if speed_factor > 1.0:
                    audio_clip = audio_clip.fx(vfx.speedx, speed_factor)
                
                clip_duration = min(audio_clip.duration, desc.get('final_clip_duration', audio_clip.duration))
                if clip_duration < audio_clip.duration:
                    audio_clip = audio_clip.subclip(0, clip_duration)
                
                # 【核心修正】使用 .fx() 來套用 volumex 效果
                audio_clip = audio_clip.fx(afx.volumex, NARRATION_VOLUME)
                audio_clip = audio_clip.set_start(desc['final_start_time'])
                narration_clips.append(audio_clip)

        original_audio = video_clip.audio
        if original_audio:
            # 【核心修正】使用 .fx() 來套用 volumex 效果
            original_audio = original_audio.fx(afx.volumex, BACKGROUND_VOLUME)
            final_audio_clips = [original_audio] + narration_clips
        else:
            final_audio_clips = narration_clips

        if not final_audio_clips:
             video_clip.audio = None
        else:
            final_audio = CompositeAudioClip(final_audio_clips)
            video_clip.audio = final_audio.set_duration(video_clip.duration)

        print(f"\n正在生成最終影片 -> {output_path}")
        video_clip.write_videofile(output_path, codec='libx264', audio_codec='aac', threads=4, logger='bar')
        
        return True

    except Exception as e:
        print(f"\n[嚴重錯誤] 影片合成過程中發生錯誤: {e}")
        traceback.print_exc()
        return False
    finally:
        # 清理 MoviePy 資源
        if video_clip: video_clip.close()
        if 'narration_clips' in locals(): 
            for clip in narration_clips: clip.close()
        if 'original_audio' in locals() and original_audio: original_audio.close()
        if 'final_audio' in locals() and 'final_audio' in locals() and final_audio: final_audio.close()


def cleanup(keyframe_dir, temp_audio_dir):
    print("正在刪除暫存檔案...")
    if os.path.isdir(keyframe_dir):
        try:
            shutil.rmtree(keyframe_dir)
            print(f" - 已刪除目錄: {keyframe_dir}")
        except Exception as e:
            print(f" - [警告] 刪除 {keyframe_dir} 失敗: {e}")
    if os.path.isdir(temp_audio_dir):
        try:
            shutil.rmtree(temp_audio_dir)
            print(f" - 已刪除目錄: {temp_audio_dir}")
        except Exception as e:
            print(f" - [警告] 刪除 {temp_audio_dir} 失敗: {e}")

def main():
    root = tk.Tk()
    root.withdraw()
    video_filepath = filedialog.askopenfilename(
        title="請選擇一個影片檔案",
        filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv")]
    )
    if not video_filepath:
        print("未選擇任何檔案，程式結束。")
        messagebox.showinfo("提示", "您沒有選擇任何影片檔案，處理程序已取消。")
        return

    VIDEO_FILENAME = os.path.basename(video_filepath)
    VIDEO_DIR = os.path.dirname(video_filepath)
    BASE_NAME = os.path.splitext(VIDEO_FILENAME)[0]
    KEYFRAME_DIR = os.path.join(VIDEO_DIR, f"{BASE_NAME}_keyframes")
    global TEMP_AUDIO_DIR
    TEMP_AUDIO_DIR = os.path.join(VIDEO_DIR, f"temp_audio_{BASE_NAME}")
    FINAL_VIDEO_PATH = os.path.join(VIDEO_DIR, f"{BASE_NAME}_narrated.mp4")
    FINAL_TXT = os.path.join(VIDEO_DIR, f"{BASE_NAME}_final_script.txt")

    video_summary = simpledialog.askstring("影片摘要", "請為這段影片提供一個簡短的摘要（例如：主角是誰、在做什麼）：", parent=root)
    if not video_summary:
        video_summary = "一段影片。"

    start_time = time.time()
    print("="*50)
    print(f"      口述影像自動生成腳本已啟動\n      處理影片: {VIDEO_FILENAME}")
    print("="*50)

    try:
        with VideoFileClip(video_filepath) as video:
            video_total_duration = video.duration
        
        # --- 完整流程 ---
        speech_segments = step0_detect_voice_segments(video_filepath)
        if not step1_extract_keyframes(video_filepath, KEYFRAME_DIR):
            raise RuntimeError("步驟 1 失敗，程式結束。")

        non_dialogue_segments = get_non_dialogue_segments(speech_segments, video_total_duration)
        initial_descriptions = step2_generate_initial_descriptions(API_KEY, KEYFRAME_DIR, video_summary, video_total_duration, non_dialogue_segments)
        if not initial_descriptions:
            raise RuntimeError("步驟 2 失敗，程式結束。")

        refined_descriptions = step3_refine_and_merge_descriptions(API_KEY, initial_descriptions, video_summary, non_dialogue_segments)
        if not refined_descriptions:
            # 步驟3失敗時，使用步驟2的結果繼續
            print("[警告] 步驟 3 失敗，將使用未精煉的描述繼續。")
            refined_descriptions = initial_descriptions

        audio_data = step4_generate_audio_and_measure_duration(refined_descriptions)
        if not audio_data:
            raise RuntimeError("步驟 4 失敗，程式結束。")

        timeline_data = step5_plan_timeline(audio_data, video_total_duration, non_dialogue_segments)
        if not timeline_data:
            print("\n[流程中止] 步驟 5 未能規劃任何旁白。")

        if timeline_data and not step6_synthesize_final_video(video_filepath, timeline_data, FINAL_VIDEO_PATH):
            raise RuntimeError("步驟 6 失敗，程式結束。")

        # 寫入文字稿
        if timeline_data:
            with open(FINAL_TXT, 'w', encoding='utf-8') as f:
                f.write(f"影片 '{VIDEO_FILENAME}' 的口述影像文字稿\n" + "="*40 + "\n\n")
                for desc in timeline_data:
                    td = timedelta(seconds=desc['final_start_time'])
                    minutes, seconds = divmod(td.seconds, 60)
                    f.write(f"[{minutes:02d}:{seconds:02d}] {desc['text']}\n")
            print(f"\n[成功] 文字稿已儲存至 {FINAL_TXT}")

        end_time = time.time()
        summary_message = (
            f"所有處理流程已成功完成！\n\n"
            f"最終影片已儲存為：\n{FINAL_VIDEO_PATH}\n\n"
            f"口述影像文字稿已儲存為：\n{FINAL_TXT}\n\n"
            f"總共耗時: {end_time - start_time:.2f} 秒"
        )
        print("\n" + "="*50)
        print(summary_message.replace("\n\n", "\n"))
        print("="*50)
        messagebox.showinfo("處理完成", summary_message)

    except Exception as e:
        error_message = f"[流程中止] 處理過程中發生嚴重錯誤: {e}"
        print(error_message)
        traceback.print_exc()
        messagebox.showerror("處理失敗", error_message)
    finally:
        if AUTO_CLEANUP_TEMP_FILES:
            cleanup(KEYFRAME_DIR, TEMP_AUDIO_DIR)
        else:
            root_cleanup = tk.Tk()
            root_cleanup.withdraw()
            if messagebox.askyesno("清理暫存", "是否要刪除過程中產生的暫存檔案？", parent=root_cleanup):
                cleanup(KEYFRAME_DIR, TEMP_AUDIO_DIR)
            else:
                print("已保留所有暫存檔案。")
            root_cleanup.destroy()

    print("\n--- 程式執行完畢 ---")

if __name__ == '__main__':
    main()