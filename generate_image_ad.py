# generate_image_ad.py (以 RAGforUIdemon 為基礎，改為單張圖片 + 使用者描述)

import os
import sys
import io
import base64
import uuid
import argparse
import traceback
import tkinter as tk
from tkinter import messagebox

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
        "套件缺失錯誤",
        f"錯誤：缺少必要的套件。請在適當環境中安裝相依。
\n詳細錯誤: {e}"
    )
    sys.exit(1)

os.environ["CHROMA_SERVER_NO_ANALYTICS"] = "True"

doc_id_to_summary_map = {}
ID_KEY = "doc_id"

# --------------------------------------------------------------------------
#                           模型與小工具
# --------------------------------------------------------------------------

def set_Model(model_path: str):
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
        return None, None


def encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def looks_like_base64(sb: str) -> bool:
    import re
    return re.match("^[A-Za-z0-9+/]+[=]{0,2}$", sb) is not None


def is_image_data(b64data: str) -> bool:
    signatures = {
        b"\xFF\xD8\xFF": "jpg",
        b"\x89\x50\x4E\x47\x0D\x0A\x1A\x0A": "png",
        b"\x47\x49\x46\x38": "gif",
        b"\x52\x49\x46\x46": "webp",
    }
    try:
        header = base64.b64decode(b64data)[:8]
        return any(header.startswith(sig) for sig in signatures)
    except Exception:
        return False

# --------------------------------------------------------------------------
#                           RAG: Retriever
# --------------------------------------------------------------------------

def create_multi_vector_retriever(vectorstore, texts, images):
    store = InMemoryStore()
    retriever = MultiVectorRetriever(vectorstore=vectorstore, docstore=store, id_key=ID_KEY)

    if not texts or not images or len(texts) != len(images):
        print("[錯誤] 圖片與文字數量不一致或為空。", file=sys.stderr)
        return None

    doc_ids = [str(uuid.uuid4()) for _ in images]
    summary_docs = [Document(page_content=s, metadata={ID_KEY: doc_ids[i]}) for i, s in enumerate(texts)]
    original_docs = [Document(page_content=content, metadata={ID_KEY: doc_ids[i]}) for i, content in enumerate(images)]

    try:
        retriever.vectorstore.add_documents(summary_docs)
        retriever.docstore.mset(list(zip(doc_ids, original_docs)))
        print(f"已成功添加 {len(images)} 個項目到檢索器。")
    except Exception as e:
        print(f"[錯誤] 添加文件到向量儲存或檔案儲存時失敗: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return None

    global doc_id_to_summary_map
    doc_id_to_summary_map = {doc_ids[i]: s for i, s in enumerate(texts)}

    return retriever


def set_DB(texts, imgs):
    try:
        embeddings_model_name = "sentence-transformers/all-MiniLM-L6-v2"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name, model_kwargs={'device': device})
        vectorstore = Chroma(collection_name=f"mm_rag_{uuid.uuid4()}", embedding_function=embeddings)
        print("向量資料庫初始化完成。")
        return create_multi_vector_retriever(vectorstore, texts, imgs)
    except Exception as e:
        print(f"[嚴重錯誤] 設定向量資料庫或檢索器時失敗: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return None


# --------------------------------------------------------------------------
#                      Llama Vision: 構建訊息
# --------------------------------------------------------------------------

def get_llama_vision_inputs_from_docs(user_question, retrieved_docs):
    all_context_messages = []
    context_images_pil = []

    if not retrieved_docs:
        print("[警告] 檢索器未返回任何相關文件。")
        rag_instruction_text = (
            f"請根據提供的上下文文本和圖片來回答問題。\n\n上下文文本：\n[無相關上下文]\n\n問題：{user_question}"
        )
        messages_for_llama = [{"role": "user", "content": [{"type": "text", "text": rag_instruction_text}]}]
        return [], messages_for_llama

    for i, doc in enumerate(retrieved_docs):
        doc_id = doc.metadata.get(ID_KEY)
        summary = doc_id_to_summary_map.get(doc_id, "[摘要遺失]")
        content = doc.page_content

        if looks_like_base64(content) and is_image_data(content):
            try:
                img_data = base64.b64decode(content)
                pil_image = Image.open(io.BytesIO(img_data)).convert("RGB")
                context_images_pil.append(pil_image)
                all_context_messages.append({"type": "image", "content": pil_image})
                all_context_messages.append({"type": "text", "text": f"圖片描述: {summary}"})
            except Exception as e:
                print(f"轉換 Base64 圖片失敗 (文件 ID: {doc_id}): {e}")
                all_context_messages.append({"type": "text", "text": f"[圖片無法加載，描述: {summary}]"})
        else:
            all_context_messages.append({"type": "text", "text": content})

    combined_text_context = "\n".join([m["text"] for m in all_context_messages if m["type"] == "text"])

    rag_instruction_text = f"""
你是專業的口述影像撰寫者。請參考下列圖片與文字重點，生成一段自然、客觀且具體的中文口述影像，避免主觀推測。

上下文：
{combined_text_context}

問題：{user_question}
"""

    final_user_content_list = []
    for img_pil in context_images_pil:
        final_user_content_list.append({"type": "image", "content": img_pil})
    final_user_content_list.append({"type": "text", "text": rag_instruction_text})

    messages_for_llama = [{"role": "user", "content": final_user_content_list}]
    return context_images_pil, messages_for_llama


# --------------------------------------------------------------------------
#                           主執行流程 (單圖)
# --------------------------------------------------------------------------

def run_single_image_narration(model_path: str, image_file: str, user_desc: str):
    model, processor = set_Model(model_path)
    if not model or not processor:
        sys.exit(1)

    if not os.path.isfile(image_file):
        print(f"[錯誤] 圖片檔不存在: {image_file}", file=sys.stderr)
        sys.exit(1)

    try:
        img_b64 = encode_image_to_base64(image_file)
    except Exception as e:
        print(f"[錯誤] 讀取或編碼圖片失敗: {e}", file=sys.stderr)
        sys.exit(1)

    texts = [user_desc.strip()]
    imgs = [img_b64]

    retriever = set_DB(texts, imgs)
    if not retriever:
        print("[錯誤] 設定 RAG 檢索器失敗。", file=sys.stderr)
        sys.exit(1)

    user_question = "請根據圖片與上述重點生成一段口述影像。"
    print(f"\n正在針對問題進行檢索: '{user_question}'")
    try:
        retrieved_docs = retriever.invoke(user_question)
    except Exception as e:
        print(f"[錯誤] 執行檢索時失敗: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        retrieved_docs = []

    llama_images, llama_messages = get_llama_vision_inputs_from_docs(user_question, retrieved_docs)

    try:
        input_text_for_processor = processor.apply_chat_template(
            llama_messages, add_generation_prompt=True, tokenize=False
        )
        inputs = processor(images=llama_images, text=input_text_for_processor, return_tensors="pt").to(model.device)
        print("模型輸入準備完成。")
    except Exception as e:
        print(f"[嚴重錯誤] 處理模型輸入時失敗: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

    print("\n正在生成答案...")
    generate_kwargs = {
        "max_new_tokens": 512,
        "do_sample": True,
        "top_p": 0.9,
        "temperature": 0.7,
        "pad_token_id": processor.tokenizer.eos_token_id,
        "eos_token_id": processor.tokenizer.eos_token_id,
    }
    try:
        output = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            **generate_kwargs
        )
        response_text = processor.decode(
            output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True
        )

        print("\n--- 模型生成的答案 ---")
        print(response_text)
        print(f"\nFINAL_IMAGE: {os.path.abspath(image_file)}")
        print(f"FINAL_ANSWER: {response_text}")

    except Exception as e:
        print(f"[嚴重錯誤] 模型生成答案時失敗: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        print("\nFINAL_ANSWER: [模型生成失敗]")
        sys.exit(1)


# --- 命令列參數解析 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用 Llama 3.2 Vision 進行單張圖片的口述影像生成 (RAG 驅動)")
    parser.add_argument("--model_path", type=str, required=True, help="Llama 模型的路徑")
    parser.add_argument("--image_file", type=str, required=True, help="單張圖片檔案路徑")
    parser.add_argument("--desc", type=str, required=True, help="使用者提供的圖片描述重點")

    args = parser.parse_args()

    if not os.path.isdir(args.model_path):
        print(f"[錯誤] 模型路徑不存在或不是資料夾: {args.model_path}", file=sys.stderr)
        sys.exit(1)

    run_single_image_narration(args.model_path, args.image_file, args.desc)
