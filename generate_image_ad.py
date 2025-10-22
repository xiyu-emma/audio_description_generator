# generate_image_ad.py (修改版：專注於單圖生成，RAG 作為風格參考)

import os
import sys
import io
import base64
import uuid
import argparse
import traceback
import tkinter as tk
from tkinter import messagebox
from typing import List, Dict, Union, Tuple, Optional
import time

# --- 核心套件載入 ---
try:
    import torch
    from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig
    from PIL import Image
    from langchain_core.documents import Document
    from langchain.retrievers import MultiVectorRetriever
    from langchain.storage import InMemoryStore
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError as e:
    root = tk.Tk()
    root.withdraw()
    messagebox.showerror(
        "套件缺失錯誤",f"錯誤：缺少必要的套件。請在適當環境中安裝相依。詳細錯誤: {e}"
    )
    sys.exit(1)

os.environ["CHROMA_SERVER_NO_ANALYTICS"] = "True"

doc_id_to_summary_map: Dict[str, str] = {} # 確保類型提示
ID_KEY = "doc_id"

# --------------------------------------------------------------------------
#                           模型與小工具 (大部分不變)
# --------------------------------------------------------------------------

def set_Model(model_path: str) -> Tuple[Optional[AutoModelForCausalLM], Optional[AutoProcessor]]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cpu":
        print("[警告] 未偵測到 CUDA GPU，將使用 CPU 載入模型，速度會較慢並可能需大量記憶體。")

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    try:
        print(f"正在從 '{model_path}' 載入模型和處理器...")
        # 增加 trust_remote_code=True 可能是某些模型載入所必需的
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True
        )
        print(f"模型 '{os.path.basename(model_path)}' 成功載入。")
        return model, processor
    except Exception as e:
        print(f"[嚴重錯誤] 載入模型或處理器失敗: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return None, None

def encode_image_to_base64(image_path: str) -> str:
    """將圖片檔案編碼為 Base64 字串"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def pil_image_to_base64(image: Image.Image, format="JPEG") -> str:
    """將 PIL Image 物件轉換為 Base64 字串"""
    buffered = io.BytesIO()
    image.save(buffered, format=format)
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def base64_to_pil_image(base64_string: str) -> Optional[Image.Image]:
    """將 Base64 字串解碼為 PIL Image 物件"""
    try:
        img_data = base64.b64decode(base64_string)
        return Image.open(io.BytesIO(img_data)).convert("RGB")
    except Exception as e:
        print(f"Base64 解碼為 PIL 圖片失敗: {e}")
        return None

def looks_like_base64(sb: str) -> bool:
    import re
    return re.match("^[A-Za-z0-9+/]+[=]{0,2}$", sb) is not None

def is_image_data(b64data: str) -> bool:
    # 保持不變
    signatures = {
        b"\xFF\xD8\xFF": "jpg", b"\x89\x50\x4E\x47\x0D\x0A\x1A\x0A": "png",
        b"\x47\x49\x46\x38": "gif", b"\x52\x49\x46\x46": "webp",
    }
    try:
        header = base64.b64decode(b64data)[:8]
        return any(header.startswith(sig) for sig in signatures)
    except Exception: return False

# --------------------------------------------------------------------------
#                           RAG: Retriever (大部分不變)
# --------------------------------------------------------------------------

def create_multi_vector_retriever(vectorstore, texts: List[str], images_b64: List[str]) -> Optional[MultiVectorRetriever]:
    """建立多向量檢索器，儲存 base64 圖片"""
    store = InMemoryStore()
    retriever = MultiVectorRetriever(vectorstore=vectorstore, docstore=store, id_key=ID_KEY)

    if not texts or not images_b64 or len(texts) != len(images_b64):
        print("[錯誤] 圖片與文字數量不一致或為空。", file=sys.stderr)
        return None

    doc_ids = [str(uuid.uuid4()) for _ in images_b64]
    # Vectorstore 儲存文字描述 (summary)
    summary_docs = [Document(page_content=s, metadata={ID_KEY: doc_ids[i]}) for i, s in enumerate(texts)]
    # Docstore 儲存原始 Base64 圖片
    original_docs = [Document(page_content=content, metadata={ID_KEY: doc_ids[i]}) for i, content in enumerate(images_b64)]

    try:
        retriever.vectorstore.add_documents(summary_docs)
        retriever.docstore.mset(list(zip(doc_ids, original_docs)))
        print(f"已成功添加 {len(images_b64)} 個項目到檢索器。")
    except Exception as e:
        print(f"[錯誤] 添加文件到向量儲存或檔案儲存時失敗: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return None

    global doc_id_to_summary_map
    doc_id_to_summary_map = {doc_ids[i]: s for i, s in enumerate(texts)}
    return retriever

def set_DB(texts: List[str], imgs_b64: List[str]) -> Optional[MultiVectorRetriever]:
    """初始化向量資料庫並建立檢索器"""
    try:
        embeddings_model_name = "sentence-transformers/all-MiniLM-L6-v2"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name, model_kwargs={'device': device})
        vectorstore = Chroma(collection_name=f"mm_rag_{uuid.uuid4()}", embedding_function=embeddings)
        print("向量資料庫初始化完成。")
        return create_multi_vector_retriever(vectorstore, texts, imgs_b64)
    except Exception as e:
        print(f"[嚴重錯誤] 設定向量資料庫或檢索器時失敗: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return None

# --------------------------------------------------------------------------
#                   Llama Vision: 構建訊息 (核心修改)
# --------------------------------------------------------------------------

def get_llama_inputs_for_single_image_narration(
    target_image_pil: Image.Image,
    target_description: str,
    retrieved_docs: List[Document]
) -> Tuple[List[Image.Image], List[Dict[str, Union[str, List[Dict]]]]]:
    """
    構建 Llama Vision 的輸入，包含目標圖片、目標描述和 RAG 範例。
    """
    all_images_pil = [target_image_pil] # 第一張永遠是目標圖片
    prompt_content_list = []

    # 1. 添加目標圖片和使用者描述到 Prompt 開頭
    prompt_content_list.append({"type": "image", "content": target_image_pil})
    prompt_content_list.append({"type": "text", "text": f"【目標圖片的重點描述】\n{target_description}\n"})

    # 2. 添加 RAG 檢索到的範例
    prompt_content_list.append({"type": "text", "text": "【風格與內容參考範例】\n（以下為資料庫中與上述描述相關的圖片及其口述影像）\n"})
    retrieved_count = 0
    if retrieved_docs:
        print(f"將使用 {len(retrieved_docs)} 個檢索到的文件作為參考範例。")
        for i, doc in enumerate(retrieved_docs):
            doc_id = doc.metadata.get(ID_KEY)
            # 從 docstore 獲取原始 base64 圖片內容
            original_image_b64 = doc.page_content
            # 從 map 獲取對應的文字描述 (範例口述影像)
            example_narration = doc_id_to_summary_map.get(doc_id, "[範例口述影像遺失]")

            # 確保內容是圖片
            if looks_like_base64(original_image_b64) and is_image_data(original_image_b64):
                example_image_pil = base64_to_pil_image(original_image_b64)
                if example_image_pil:
                    all_images_pil.append(example_image_pil)
                    prompt_content_list.append({"type": "text", "text": f"\n--- 範例 {i+1} ---"})
                    prompt_content_list.append({"type": "image", "content": example_image_pil})
                    prompt_content_list.append({"type": "text", "text": f"範例 {i+1} 的口述影像:\n{example_narration}"})
                    retrieved_count += 1
                else:
                    print(f"警告：無法將檢索到的文件 ID {doc_id} 的 Base64 轉換為圖片。")
            else:
                 print(f"警告：檢索到的文件 ID {doc_id} 的內容不是有效的 Base64 圖片。")

    if retrieved_count == 0:
        prompt_content_list.append({"type": "text", "text": "(未找到相關範例)\n"})

    # 3. 添加最終指令
    final_instruction = f"""
---
【你的任務】
作為一位專業的口述影像撰寫者，請專注於描述最上方提供的【目標圖片】。
參考【目標圖片的重點描述】作為內容基礎，並學習【風格與內容參考範例】中口述影像的客觀性、詳細程度和流暢自然的語氣。
生成一段高品質的中文口述影像，僅描述目標圖片的視覺內容，避免加入範例圖片的內容或進行主觀臆測。
"""
    prompt_content_list.append({"type": "text", "text": final_instruction})

    # 4. 構建成 Llama 需要的格式
    messages_for_llama = [{"role": "user", "content": prompt_content_list}]

    return all_images_pil, messages_for_llama

# --------------------------------------------------------------------------
#                           主執行流程 (單圖生成)
# --------------------------------------------------------------------------

def run_single_image_narration(model_path: str, image_file: str, user_desc: str):
    """執行單張圖片的口述影像生成流程"""
    model, processor = set_Model(model_path)
    if not model or not processor:
        sys.exit(1)

    # --- 載入目標圖片 ---
    try:
        target_image_pil = Image.open(image_file).convert("RGB")
    except FileNotFoundError:
        print(f"[錯誤] 目標圖片檔不存在: {image_file}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"[錯誤] 讀取目標圖片失敗: {e}", file=sys.stderr)
        sys.exit(1)

    # --- 建立 RAG 資料庫 (從 data/ 子資料夾) ---
    def _read_text_file(path: str) -> str:
        # (保持不變)
        try: return open(path, "r", encoding="utf-8").read()
        except UnicodeDecodeError:
            try: return open(path, "r", encoding="utf-8-sig").read()
            except Exception: return open(path, "r", encoding="gbk", errors="ignore").read()

    def _load_pairs_from_data_dirs() -> Tuple[List[str], List[str]]:
        # (保持不變)
        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(base_dir, "data")
        img_dir = os.path.join(data_dir, "source_images")
        txt_dir = os.path.join(data_dir, "source_texts")

        if not os.path.isdir(img_dir) or not os.path.isdir(txt_dir):
            print("[警告] 找不到 data/source_images 或 data/source_texts，RAG 將無範例可用。")
            return [], []

        text_map = {}
        for fn in os.listdir(txt_dir):
            if fn.lower().endswith(".txt"):
                stem = os.path.splitext(fn)[0]
                fp = os.path.join(txt_dir, fn)
                try: text_map[stem] = _read_text_file(fp).strip()
                except Exception as e: print(f"[警告] 讀取文字檔失敗: {fp}，{e}")

        exts = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
        texts_db, imgs_db_b64 = [], []
        for fn in sorted(os.listdir(img_dir)):
            if fn.lower().endswith(exts):
                stem = os.path.splitext(fn)[0]
                if stem in text_map:
                    img_fp = os.path.join(img_dir, fn)
                    try:
                        imgs_db_b64.append(encode_image_to_base64(img_fp))
                        texts_db.append(text_map[stem])
                    except Exception as e: print(f"[警告] 載入或編碼圖片失敗: {img_fp}，{e}")
                else: print(f"[警告] 找不到與圖片對應的文字檔：{fn}，已略過。")
        print(f"已從資料夾載入 {len(imgs_db_b64)} 組圖片/文字配對建立 RAG 資料庫。")
        return texts_db, imgs_db_b64

    data_texts, data_imgs_b64 = _load_pairs_from_data_dirs()

    retriever = set_DB(data_texts, data_imgs_b64)
    # 注意：這次不需要將當前圖片加入 DB

    # --- 使用使用者描述進行 RAG 檢索 ---
    if retriever:
        retrieval_query = user_desc.strip()
        print(f"\n正在根據您的描述進行檢索以尋找參考範例: '{retrieval_query}'")
        try:
            # 檢索相似的文字描述，返回的是 Document(page_content=base64_image, metadata={'doc_id': ...})
            retrieved_docs = retriever.invoke(retrieval_query)
        except Exception as e:
            print(f"[警告] 執行 RAG 檢索時失敗: {e}", file=sys.stderr)
            retrieved_docs = []
    else:
        print("[警告] RAG 檢索器未成功建立，將不使用參考範例。")
        retrieved_docs = []

    # --- 構建 Llama 輸入 ---
    # 【重要】傳入目標圖片的 PIL 物件和使用者描述
    llama_images, llama_messages = get_llama_inputs_for_single_image_narration(
        target_image_pil, user_desc.strip(), retrieved_docs
    )

    # --- 處理輸入並生成 ---
    try:
        input_text_for_processor = processor.apply_chat_template(
            llama_messages, add_generation_prompt=True, tokenize=False
        )
        # 【重要】傳入所有圖片 (目標圖片 + 範例圖片)
        inputs = processor(images=llama_images, text=input_text_for_processor, return_tensors="pt").to(model.device)
        print("模型輸入準備完成。")
    except Exception as e:
        print(f"[嚴重錯誤] 處理模型輸入時失敗: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

    print("\n正在生成口述影像...")
    generate_kwargs = {
        "max_new_tokens": 512, "do_sample": True, "top_p": 0.9,
        "temperature": 0.1,
        "pad_token_id": processor.tokenizer.eos_token_id,
        "eos_token_id": processor.tokenizer.eos_token_id,
    }
    try:
        output = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            # pixel_values 不再需要
            **generate_kwargs
        )
        response_text = processor.decode(
            output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True
        ).strip() # 去除前後空白

        print("\n--- 模型生成的口述影像 ---")
        print(response_text)
        # 使用特定標記輸出最終答案
        print(f"\nFINAL_IMAGE: {os.path.abspath(image_file)}") # 標記原始圖片路徑
        print(f"FINAL_ANSWER: {response_text}")

    except Exception as e:
        print(f"[嚴重錯誤] 模型生成答案時失敗: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        print("\nFINAL_ANSWER: [模型生成失敗]")
        sys.exit(1)

# --- 命令列參數解析 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用 Llama 3.2 Vision 進行單張圖片的口述影像生成 (RAG 輔助)")
    parser.add_argument("--model_path", type=str, required=True, help="Llama 模型的路徑")
    # 修改為接收單一圖片檔案
    parser.add_argument("--image_file", type=str, required=True, help="要生成口述影像的單張圖片檔案路徑")
    # 修改為接收使用者描述
    parser.add_argument("--desc", type=str, required=True, help="使用者提供的關於該圖片的初步描述或重點")

    args = parser.parse_args()

    if not os.path.isdir(args.model_path):
        print(f"[錯誤] 模型路徑不存在或不是資料夾: {args.model_path}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(args.image_file):
        print(f"[錯誤] 圖片檔案不存在: {args.image_file}", file=sys.stderr)
        sys.exit(1)

    start_time = time.time()
    run_single_image_narration(args.model_path, args.image_file, args.desc)
    end_time = time.time()
    print(f"\n--- 程式執行完畢，總耗時: {end_time - start_time:.2f} 秒 ---")