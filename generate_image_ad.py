# generate_image_ad.py 

import os
import cv2
import numpy as np
import time
import random
import shutil
import asyncio
import nest_asyncio
import re
import base64
import io
import uuid
import argparse # 新增：用於解析命令列參數
import sys     # 新增：用於退出程式
from datetime import timedelta
from typing import List, Tuple
from tkinter import messagebox # 保留用於顯示初始錯誤
import tkinter as tk         # 保留用於顯示初始錯誤
import traceback

# --- 核心套件載入 ---
try:
    import torch
    from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig
    import requests
    from PIL import Image
    from langchain_core.documents import Document
    from langchain.retrievers import MultiVectorRetriever
    from langchain.storage import InMemoryStore
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings

except ImportError as e:
    # 如果缺少套件，顯示錯誤訊息 (雖然主程式可能看不到，但保留以防單獨執行)
    root = tk.Tk()
    root.withdraw()
    messagebox.showerror(
        "套件缺失錯誤",
        f"錯誤：缺少必要的套件。請確保已在 'ad_image_env' 環境中安裝所有依賴項。\n\n詳細錯誤: {e}"
    )
    sys.exit(1) # 以非零狀態退出

nest_asyncio.apply()
os.environ["CHROMA_SERVER_NO_ANALYTICS"] = "True"

# --------------------------------------------------------------------------
#                           模型和資料路徑 (由參數傳入)
# --------------------------------------------------------------------------
# 這些將由 argparse 設定，不再硬編碼
# model_path = "./models/Llama-3.2-11B-Vision-Instruct"
# image_folder_path = "./data/source_images"
# text_folder_path = "./data/source_texts"
# user_question = "總共有多少人？"

# 全域變數，用於儲存 doc_id 到摘要的映射
doc_id_to_summary_map = {}
id_key = "doc_id" # 保持一致

# --------------------------------------------------------------------------
#                           輔助函式 (基本不變)
# --------------------------------------------------------------------------

def set_Model(model_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cpu":
        print("[警告] 未偵測到 CUDA GPU，將使用 CPU 載入模型，速度會非常慢且可能因記憶體不足而失敗。")

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    try:
        print(f"正在從 '{model_path}' 載入模型和處理器...")
        processor = AutoProcessor.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quantization_config,
            device_map="auto"
        )
        print(f"模型 '{os.path.basename(model_path)}' 成功載入。")
        return model, processor
    except Exception as e:
        print(f"[嚴重錯誤] 載入模型或處理器失敗: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return None, None # 返回 None 表示失敗

def load_imgs_texts_from_folder(image_folder_path, text_folder_path):
    img_base64_list = []
    text_list = []
    img_filenames = [] # 記錄成功載入的圖片檔名
    
    try:
        img_files = sorted([f for f in os.listdir(image_folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    except FileNotFoundError:
        print(f"[錯誤] 找不到圖片資料夾: {image_folder_path}", file=sys.stderr)
        return [], [], []
        
    img_base64_map = {}
    print(f"在 '{image_folder_path}' 中找到 {len(img_files)} 個圖片檔。")
    for filename in img_files:
        file_path = os.path.join(image_folder_path, filename)
        try:
            with open(file_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                img_base64_map[filename] = encoded_string
                # img_base64_list.append(encoded_string) # 稍後根據文字檔匹配添加
                # img_filenames.append(filename) # 記錄檔名
        except Exception as e:
            print(f"無法載入或編碼圖片 '{filename}': {e}")
            
    try:
        text_files = sorted([f for f in os.listdir(text_folder_path) if f.lower().endswith('.txt')])
    except FileNotFoundError:
        print(f"[錯誤] 找不到文字資料夾: {text_folder_path}", file=sys.stderr)
        return [], [], []
        
    text_content_map = {}
    print(f"在 '{text_folder_path}' 中找到 {len(text_files)} 個文字檔。")
    successful_pairs = 0
    for filename in text_files:
        file_path = os.path.join(text_folder_path, filename)
        img_filename_key = os.path.splitext(filename)[0] # 去掉 .txt
        
        # 尋找對應的圖片檔 (忽略副檔名比較)
        corresponding_img_file = None
        for img_file in img_base64_map.keys():
            if os.path.splitext(img_file)[0] == img_filename_key:
                corresponding_img_file = img_file
                break
                
        if corresponding_img_file and corresponding_img_file in img_base64_map:
            try:
                with open(file_path, "r", encoding="utf-8") as text_file:
                    text_string = text_file.read()
            except UnicodeDecodeError:
                try:
                    with open(file_path, "r", encoding="gbk") as text_file:
                        text_string = text_file.read()
                except Exception as e:
                    print(f"無法以 UTF-8 或 GBK 載入文字檔案 '{filename}': {e}")
                    continue # 跳過這個檔案
            except Exception as e:
                print(f"無法載入文字檔案 '{filename}': {e}")
                continue # 跳過這個檔案
                
            # 只有當圖片和文字都成功載入時才加入列表
            text_list.append(text_string)
            img_base64_list.append(img_base64_map[corresponding_img_file])
            img_filenames.append(corresponding_img_file)
            successful_pairs += 1
        else:
            print(f"警告：文字檔 '{filename}' 找不到對應的圖片檔。")

    print(f"成功匹配並載入 {successful_pairs} 組圖片與文字。")
    return img_base64_list, text_list, img_filenames


def looks_like_base64(sb):
    return re.match("^[A-Za-z0-9+/]+[=]{0,2}$", sb) is not None

def is_image_data(b64data):
    image_signatures = {
        b"\xFF\xD8\xFF": "jpg", b"\x89\x50\x4E\x47\x0D\x0A\x1A\x0A": "png",
        b"\x47\x49\x46\x38": "gif", b"\x52\x49\x46\x46": "webp",
    }
    try:
        header = base64.b64decode(b64data)[:8]
        for sig in image_signatures:
            if header.startswith(sig): return True
        return False
    except Exception: return False

def resize_base64_image(base64_string, size=(128, 128)):
    try:
        img_data = base64.b64decode(base64_string)
        img = Image.open(io.BytesIO(img_data))
        resized_img = img.resize(size, Image.LANCZOS)
        buffered = io.BytesIO()
        resized_img.save(buffered, format=img.format if img.format else "PNG") # 提供預設格式
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
    except Exception as e:
        print(f"調整圖片大小失敗: {e}")
        return None

# --------------------------------------------------------------------------
#                           RAG 核心函式
# --------------------------------------------------------------------------

def create_multi_vector_retriever(vectorstore, texts, images):
    store = InMemoryStore()
    retriever = MultiVectorRetriever(vectorstore=vectorstore, docstore=store, id_key=id_key)

    if not texts or not images or len(texts) != len(images):
        print("[錯誤] 圖片和文字數量不匹配或為空，無法建立檢索器。", file=sys.stderr)
        return None
        
    doc_ids = [str(uuid.uuid4()) for _ in images]
    summary_docs = [Document(page_content=s, metadata={id_key: doc_ids[i]}) for i, s in enumerate(texts)]
    original_docs = [Document(page_content=content, metadata={id_key: doc_ids[i]}) for i, content in enumerate(images)]
    
    try:
        retriever.vectorstore.add_documents(summary_docs)
        retriever.docstore.mset(list(zip(doc_ids, original_docs)))
        print(f"已成功添加 {len(images)} 個項目到檢索器。")
    except Exception as e:
        print(f"[錯誤] 添加文件到向量儲存或文件儲存時失敗: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return None

    global doc_id_to_summary_map
    doc_id_to_summary_map = {doc_ids[i]: s for i, s in enumerate(texts)}
    
    # 移除 IPython.display 相關程式碼
    print("\n--- 檢查前 2 個綁定的圖片與圖片描述 ---")
    for i, doc_id in enumerate(doc_ids[:2]):
        summary = texts[i]
        original_content = images[i]
        print(f"\n綁定項目 {i+1} (ID: {doc_id}):")
        print(f"  **圖片描述 (摘要):** {summary[:100]}...")
        if looks_like_base64(original_content) and is_image_data(original_content):
            print(f"  **原始圖片:** (Base64 長度: {len(original_content)} 字元)")
        else:
             print(f"  **原始內容 (非圖片):** {original_content[:100]}...")
        print("-" * 40)
    print("--- 綁定檢查結束 ---")
    
    return retriever

def set_DB(texts, imgs):
    try:
        embeddings_model_name = "sentence-transformers/all-MiniLM-L6-v2"
        # 嘗試使用 CUDA，如果不可用則回退到 CPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name, model_kwargs={'device': device})
        
        # 使用記憶體中的 ChromaDB，不進行持久化，避免權限問題
        vectorstore = Chroma(
            collection_name=f"mm_rag_{uuid.uuid4()}", # 使用唯一名稱避免衝突
            embedding_function=embeddings
        )
        print("向量資料庫初始化完成。")
        
        retriever = create_multi_vector_retriever(vectorstore, texts, imgs)
        return retriever
    except Exception as e:
        print(f"[嚴重錯誤] 設定向量資料庫或檢索器時失敗: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return None

def get_llama_vision_inputs_from_docs(user_question, retrieved_docs):
    all_context_messages = []
    context_images_pil = []

    if not retrieved_docs:
        print("[警告] 檢索器未返回任何相關文件。")
        # 即使沒有檢索到文件，也要回傳符合格式的空列表
        rag_instruction_text = f"根據提供的上下文文本和圖片來回答問題。\n\n上下文文本：\n[無相關上下文]\n\n問題：{user_question}"
        messages_for_llama = [{"role": "user", "content": [{"type": "text", "text": rag_instruction_text}]}]
        return [], messages_for_llama

    print(f"檢索到 {len(retrieved_docs)} 個相關文件。")
    for i, doc in enumerate(retrieved_docs):
        doc_id = doc.metadata.get(id_key)
        summary = doc_id_to_summary_map.get(doc_id, "[摘要遺失]")
        content = doc.page_content

        if looks_like_base64(content) and is_image_data(content):
            try:
                img_data = base64.b64decode(content)
                pil_image = Image.open(io.BytesIO(img_data)).convert("RGB")
                context_images_pil.append(pil_image)
                # 添加圖片和對應的摘要
                all_context_messages.append({"type": "image", "content": pil_image})
                all_context_messages.append({"type": "text", "text": f"圖片 {i+1} 的描述: {summary}"})
            except Exception as e:
                print(f"轉換 Base64 圖片失敗 (文件 ID: {doc_id}): {e}")
                all_context_messages.append({"type": "text", "text": f"[圖片 {i+1} 無法加載，描述: {summary}]"})
        else:
            # 如果是文本內容 (理論上不應發生，因為我們存的是圖片)
            all_context_messages.append({"type": "text", "text": content})

    # 構建給 Llama 模型的提示
    # 將所有文字內容串接起來
    combined_text_context = "\n".join([msg["text"] for msg in all_context_messages if msg["type"] == "text"])

    rag_instruction_text = f"""請根據以下提供的上下文圖片和文字描述來回答問題。

上下文:
{combined_text_context}

問題: {user_question}
"""
    # 構建 Llama message
    final_user_content_list = []
    for img_pil in context_images_pil:
        final_user_content_list.append({"type": "image", "content": img_pil})
    final_user_content_list.append({"type": "text", "text": rag_instruction_text})

    messages_for_llama = [{"role": "user", "content": final_user_content_list}]

    return context_images_pil, messages_for_llama

# --------------------------------------------------------------------------
#                           主執行流程
# --------------------------------------------------------------------------

def main(model_path, image_folder_path, text_folder_path, user_question):
    """主執行函式"""
    
    # **1 配置模型
    model, processor = set_Model(model_path)
    if not model or not processor:
        sys.exit(1) # 模型載入失敗，退出

    # **2 讀資料
    imgs, texts, _ = load_imgs_texts_from_folder(image_folder_path, text_folder_path)
    if not imgs or not texts:
        print("[錯誤] 未能成功載入圖片或文字資料，程式中止。", file=sys.stderr)
        sys.exit(1)

    # **3 設定 RAG 檢索器 (資料庫)
    retriever = set_DB(texts, imgs)
    if not retriever:
        print("[錯誤] 設定 RAG 檢索器失敗，程式中止。", file=sys.stderr)
        sys.exit(1)

    # **4 進行檢索
    print(f"\n正在針對問題進行檢索: '{user_question}'")
    try:
        retrieved_docs = retriever.invoke(user_question)
    except Exception as e:
        print(f"[錯誤] 執行檢索時失敗: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        retrieved_docs = [] # 出錯時給空列表

    # **5 準備模型輸入
    llama_images, llama_messages = get_llama_vision_inputs_from_docs(user_question, retrieved_docs)

    print(f"準備發送給模型的圖片數量: {len(llama_images)} 張。")
    # print(f"準備發送給模型的訊息結構: {llama_messages}") # 訊息可能很長，選擇性打印

    # **6 格式化輸入
    try:
        input_text_for_processor = processor.apply_chat_template(
            llama_messages, add_generation_prompt=True, tokenize=False
        )
        inputs = processor(
            images=llama_images, text=input_text_for_processor, return_tensors="pt"
        ).to(model.device)
        print("模型輸入準備完成。")
    except Exception as e:
        print(f"[嚴重錯誤] 處理模型輸入時失敗: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

    # **7 模型生成答案
    print("\n正在生成答案...")
    generate_kwargs = {
        "max_new_tokens": 1024, "do_sample": True, "top_p": 0.9,
        "temperature": 0.7, "pad_token_id": processor.tokenizer.eos_token_id,
        "eos_token_id": processor.tokenizer.eos_token_id,
    }
    try:
        output = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            #pixel_values=inputs.get("pixel_values"), # 如果有 pixel_values 則傳入
            **generate_kwargs
        )
        response_text = processor.decode(
            output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True
        )
        
        # **8 輸出最終結果
        print("\n--- 模型生成的答案 ---")
        print(response_text)
        
        # 使用特定標記輸出最終答案，方便主程式擷取
        print(f"\nFINAL_ANSWER: {response_text}")

    except Exception as e:
        print(f"[嚴重錯誤] 模型生成答案時失敗: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        # 即使生成失敗，也打印一個標記，告知主程式
        print("\nFINAL_ANSWER: [模型生成失敗]")
        sys.exit(1)

# --- 命令列參數解析 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用 Llama 3.2 Vision 進行 RAG 問答")
    parser.add_argument("--model_path", type=str, required=True, help="Llama 模型的路徑")
    parser.add_argument("--image_path", type=str, required=True, help="圖片資料夾的路徑")
    parser.add_argument("--text_path", type=str, required=True, help="文字描述資料夾的路徑")
    parser.add_argument("--question", type=str, required=True, help="要詢問模型的問題")
    
    args = parser.parse_args()
    
    # 檢查路徑是否存在
    if not os.path.isdir(args.model_path):
        print(f"[錯誤] 模型路徑不存在或不是一個資料夾: {args.model_path}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isdir(args.image_path):
        print(f"[錯誤] 圖片路徑不存在或不是一個資料夾: {args.image_path}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isdir(args.text_path):
        print(f"[錯誤] 文字路徑不存在或不是一個資料夾: {args.text_path}", file=sys.stderr)
        sys.exit(1)
        
    start_time = time.time()
    main(args.model_path, args.image_path, args.text_path, args.question)
    end_time = time.time()
    print(f"\n--- 程式執行完畢，總耗時: {end_time - start_time:.2f} 秒 ---")