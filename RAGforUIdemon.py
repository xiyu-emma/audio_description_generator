#1. 啟動模型及更改配置
import torch
from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig
import requests
from PIL import Image
import os
import gc
import torch

# 1. 指定模型路徑或名稱
model_name = r"models\Llama-3.2-11B-Vision-Instruct" # 假設的模型ID，請替換為實際的Hugging Face模型ID或本地路徑
def set_Model():       
    # 檢查是否有可用的 GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 2. 載入 Processor 和 Model 開機處理
    # === 新增部分：定義 4-bit 量化配置 ===
    # BitsAndBytesConfig 是用於配置 4-bit 量化的關鍵
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True, # 啟用 4-bit 量化
        bnb_4bit_quant_type="nf4", # 量化類型：nf4 是 NormalFloat 4-bit，通常推薦使用
        bnb_4bit_compute_dtype=torch.bfloat16, # 計算時使用的數據類型，通常保持為 bfloat16 或 float16 以保持一定精度
        bnb_4bit_use_double_quant=True, # 啟用雙重量化，可以進一步減少記憶體佔用，但速度可能稍慢
    )
    try:
        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(#10G
            model_name, 
            quantization_config=quantization_config, # 傳入量化配置
            device_map="auto" # 自動分配到可用設備，對於量化模型通常是最佳實踐
        )
        #model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(device)#device_map="auto" #22G 自動將模型層分配到可用的設備，也可以直接 .to(device)
        #print(f"Model '{model_name}' loaded successfully on {device}.")
        return model, processor #回傳模型
    
    except Exception as e:
        print(f"Error loading model or processor: {e}")
        exit()
model, processor = set_Model()#**1配置模型----------------------------------------------------------------------------------------------------------------



# 3. 準備輸入數據 我原本是用UnstructuredLoader自動切但它壞了你們能嘗試，取代方案是用若瑄的自動切程式
#from langchain_unstructured import UnstructuredLoader # 注意這裡變成了 UnstructuredLoader
#舊from langchain_community.document_loaders import UnstructuredFileLoader

import base64
def load_imgs_texts_from_folder(image_folder_path, text_folder_path):
    img_base64_list = []
    text_list = []
    # 1. 載入圖片 (Base64 編碼)
    img_files = sorted([f for f in os.listdir(image_folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])#按檔名排序
    img_base64_map = {} # 用字典儲存，方便按文件名匹配
    for filename in img_files: # 遍歷每個圖片檔案
        file_path = os.path.join(image_folder_path, filename) # 獲取完整的檔案路徑
        try:
            with open(file_path, "rb") as image_file: # 以二進位讀取模式打開圖片檔案
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8') # 讀取檔案內容並 Base64 編碼，然後解碼為 UTF-8 字串
                img_base64_map[filename] = encoded_string # 將 Base64 字串存入字典，鍵為檔名
                img_base64_list.append(encoded_string) # 將 Base64 字串添加到列表中
        except Exception as e: # 捕獲任何載入或編碼時的錯誤
            print(f"無法載入或編碼圖片 '{filename}': {e}") # 印出錯誤訊息
            
    # 2. 載入文字 (.txt 檔案)
    text_files = sorted([f for f in os.listdir(text_folder_path) if f.lower().endswith('.txt')])#按檔名排序
    text_content_map = {} # 用字典儲存文字內容，方便按檔名匹配
    for filename in text_files: # 遍歷每個文字檔案
        file_path = os.path.join(text_folder_path, filename) # 獲取完整的檔案路徑
        try:
            with open(file_path, "r", encoding="utf-8") as text_file: # 以 UTF-8 編碼讀取模式打開文字檔案
                text_string = text_file.read() # 讀取檔案所有內容
                text_content_map[filename] = text_string # 將文字內容存入字典，鍵為檔名
                text_list.append(text_string) # 將文字內容添加到列表中
        except UnicodeDecodeError: # 如果 UTF-8 解碼失敗
            try: # 嘗試使用 GBK 編碼重新讀取
                with open(file_path, "r", encoding="gbk") as text_file:
                    text_string = text_file.read()
                    text_content_map[filename] = text_string
                    text_list.append(text_string)
            except Exception as e: # 捕獲 GBK 解碼或其他錯誤
                print(f"無法載入或解碼文字檔案 '{filename}': {e}") # 印出錯誤訊息
        except Exception as e: # 捕獲其他載入錯誤
            print(f"無法載入文字檔案 '{filename}': {e}") # 印出錯誤訊息
            
    return img_base64_list, text_list # 返回圖片 Base64 列表和文字內容列表
    
FILEPATH = r"C:\Users\S207-A5000\Desktop\TYS\RAG3.2_vision\印象派畫作配對.pdf"
source_img_path = r"C:\Users\TYS\Desktop\RAG_UI\印象派畫"#需自備圖及txt的資料夾
source_txt_path = r"C:\Users\TYS\Desktop\RAG_UI\描述文字"
def set_ImgsAndTexts():
    # --- 主要 RAG 流程 ---
    if model and processor: # 只有模型成功加載才繼續
        # 3. 載入並解析文件元素
        try:
            texts = []
            imgs = []
            imgs, texts= load_imgs_texts_from_folder(source_img_path, source_txt_path)
            #print(f"成功載入 {len(imgs)} /{len(texts)}分別元素。")
    
            if texts:
                print("摘要列表不為空")
                # for i, summary_item in enumerate(texts):
                #     if i>5:
                #         break
                #     print(f"摘要 {i+1} (長度: {len(summary_item)}): {summary_item[:100]}...") # 顯示摘要前100個字元
            else:
                print("摘要列表 (texts) 為空。")
            
            if not imgs:
                print("警告：未從文件中載入任何元素。請檢查文件路徑和內容。")
            return imgs, texts
        except Exception as e:
            print(f"載入文件時發生錯誤: {e}")
            imgs = []
            
imgs, texts = set_ImgsAndTexts()#**2讀資料準備建DB--------------------------------------------------------------------------------------------------------------
        
import io
import re
import base64 # 確保有導入 Base64
from PIL import Image
from IPython.display import HTML, display, Image as IPythonImage # 導入 IPython.display 的 Image
import os # 導入 os 模組，用於文件系統操作
import uuid # 導入 uuid 模組，用於生成唯一識別碼
from langchain_core.documents import Document # 從 langchain_core 導入 Document 類，用於表示文檔
from langchain.retrievers import MultiVectorRetriever # 導入 LangChain 的 MultiVectorRetriever
from langchain.storage import InMemoryStore # 導入 LangChain 的 InMemoryStore，用於記憶體儲存
#from langchain_community.vectorstores import Chroma #舊的
from langchain_chroma import Chroma#新的
#from langchain_community.embeddings import HuggingFaceEmbeddings舊版
from langchain_huggingface import HuggingFaceEmbeddings #新版

os.environ["CHROMA_SERVER_NO_ANALYTICS"] = "True"#禁用ChromaDB 預設會嘗試收集一些匿名使用數據
def looks_like_base64(sb): # 檢查字串是否符合 Base64 格式的樣式
    return re.match("^[A-Za-z0-9+/]+[=]{0,2}$", sb) is not None # 使用正則表達式匹配 Base64 字元集和可能的填充字符

def is_image_data(b64data): # 檢查 Base64 編碼的資料是否為圖片資料
    image_signatures = { # 定義常見圖片格式的檔案簽名 (魔術數字)
        b"\xFF\xD8\xFF": "jpg",                      # JPEG 圖片簽名
        b"\x89\x50\x4E\x47\x0D\x0A\x1A\x0A": "png", # PNG 圖片簽名
        b"\x47\x49\x46\x38": "gif",                  # GIF 圖片簽名
        b"\x52\x49\x46\x46": "webp",                 # WEBP 圖片簽名 (RIFF 開頭，後續還有特定格式)
    }
    try:
        header = base64.b64decode(b64data)[:8] # 解碼 Base64 資料並取出前 8 個位元組作為標頭
        for sig, format in image_signatures.items(): # 遍歷所有已知的圖片簽名
            if header.startswith(sig): # 如果標頭以某個簽名開頭
                return True # 則判斷為圖片資料
        return False # 如果沒有匹配到任何圖片簽名，則不是圖片資料
    except Exception: # 捕獲解碼或處理過程中的任何錯誤
        return False # 發生錯誤則判斷為非圖片資料

def resize_base64_image(base64_string, size=(128, 128)): # 調整 Base64 編碼圖片的大小
    img_data = base64.b64decode(base64_string) # 將 Base64 字串解碼回二進位圖片資料
    img = Image.open(io.BytesIO(img_data)) # 從記憶體中的二進位流打開圖片
    resized_img = img.resize(size, Image.LANCZOS) # 使用 LANCZOS 濾波器調整圖片大小
    buffered = io.BytesIO() # 創建一個記憶體中的二進位緩衝區
    resized_img.save(buffered, format=img.format) # 將調整大小後的圖片保存到緩衝區，格式與原始圖片相同
    return base64.b64encode(buffered.getvalue()).decode("utf-8") # 將緩衝區內容 Base64 編碼並解碼為 UTF-8 字串返回

def create_multi_vector_retriever(vectorstore, texts, images):#vectorstore, text_summaries, texts, table_summaries, tables, image_summaries, images
    #def add_documents(retriever, doc_summaries, doc_contents):
    def add_documents(retriever, doc_texts, doc_images):
        if not doc_texts or not doc_images:# 如果沒有內容，就跳過
            return
        doc_ids = [str(uuid.uuid4()) for _ in doc_images]#為每個原始內容生成一個唯一的ID
        summary_docs = [# 創建Document物件，並嵌入唯一ID與摘要
            Document(page_content=s, metadata={id_key: doc_ids[i]})
            for i, s in enumerate(doc_texts)
        ]
        # 創建原始內容的 Document 物件，新加除錯0729
        # 這些 Document 物件將被儲存到 docstore，並在檢索時返回
        original_docs = [
            Document(page_content=content, metadata={id_key: doc_ids[i]})
            for i, content in enumerate(doc_images)
        ]
        retriever.vectorstore.add_documents(summary_docs)#進入vectorstore，給Embeddings
        retriever.docstore.mset(list(zip(doc_ids, original_docs)))#原始資料與鍵值綁定,新加除錯0729把doc_images改成original_docs
        #print(f"已添加 {len(doc_images)} 個內容到檢索器。")

        #0730圖片uid map圖片摘要
        for i, doc_id in enumerate(doc_ids):
            doc_id_to_summary_map[doc_id] = doc_texts[i]
        #print(doc_id_to_summary_map)#是9-23
        #新加除錯0729
        #print("\n--- 正在輸出每個綁定的圖片與圖片描述 ---")
        for i, doc_id in enumerate(doc_ids):
            if i==2:
                break
            summary = doc_texts[i]
            original_content = doc_images[i] # 這是原始的 Base64 圖片字串
            #print(f"\n綁定項目 {i+1} (ID: {doc_id}):")
            #print(f"  **圖片描述 (摘要):** {summary}")
            if looks_like_base64(original_content) and is_image_data(original_content):
                try:
                    # 這裡我們將圖片臨時縮小到 400x300 來顯示，不影響原始儲存的圖片大小
                    temp_resized_b64 = resize_base64_image(original_content, size=(400, 300))
                    display(IPythonImage(data=base64.b64decode(temp_resized_b64)))
                    #print(f"    (Base64 長度: {len(original_content)} 字元)")
                except Exception as e:
                    print(f"    無法顯示圖片: {e}")
                    print(f"    Base64 內容預覽: {original_content[:100]}...") # 顯示部分Base64字串
            else:
                print(f"  **原始內容 (非圖片):**")
                if len(original_content) > 200:
                    print(f"    {original_content[:200]}...") # 顯示部分文本
                else:
                    print(f"    {original_content}")
            print("-" * 40)
        print("--- 綁定輸出結束 ---")    
    # Initialize the storage layer
    store = InMemoryStore()#暫存原始資料於記憶體
    id_key = "doc_id"#摘要指向原始資料
    # Create the multi-vector retriever
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=store,
        id_key=id_key,
        #k=10, # <-- 在這裡設定 k 值 最相關
    )
    add_documents(retriever, texts, images)
    return retriever

# 0730未來資料庫位置定義向量資料庫的持久化路徑
persist_directory = "./chroma_db" # 這裡假設你的向量資料庫會持久化到這個路徑

# 如果資料庫存在，先刪除它以確保每次運行都是全新的
if os.path.exists(persist_directory):
    import shutil
    shutil.rmtree(persist_directory)
    print(f"已刪除舊的向量資料庫：{persist_directory}")
doc_id_to_summary_map = {}

def set_DB():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model_kwargs = {'device': 'cuda'} # 如果你有CUDA並且想使用GPU，就改成 'cuda'，否則保持 'cpu'
    # --- 初始化嵌入模型 ---
    embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)
    # The vectorstore to use to index the summaries
    vectorstore = Chroma(
        collection_name="mm_rag_mistral",
        embedding_function=embeddings,
        #persist_directory=persist_directory # 設定持久化路徑 0730未來資料庫位置
    )
    #print("向量資料庫初始化完成。")
    # 這個字典將儲存 {doc_id: 摘要文本} 的映射，以便後續檢索摘要
    
    # Create retriever
    retriever_multi_vector_img = create_multi_vector_retriever(
        vectorstore,
        #text_summaries,
        texts,
        #table_summaries,
        #tables,
        #image_summaries,
        imgs,
    )
    return retriever_multi_vector_img

retriever_multi_vector_img = set_DB()#**3設定DB--------------------------------------------------------------------------------------------------------------------------

#5. 將使用者問題及檢索結果做成所需格式，目前能跑，但我想大改這裡
def get_llama_vision_inputs_from_docs(user_question, retrieved_docs):
#模型所需的 'messages' 格式。返回一個 PIL Image 物件列表和一個格式化的 messages 列表。
    all_context_messages = [] # 儲存所有上下文的訊息
    context_images_pil = [] # 儲存 PIL Image 物件
    id_key = "doc_id"
    for i, doc in enumerate(retrieved_docs):
        original_content_doc_id = doc.metadata.get(id_key) #從我們保存的映射字典中獲取摘要
        associated_summary = doc_id_to_summary_map.get(original_content_doc_id, "N/A (摘要未找到)")
        doc_content = doc.page_content # 這是原始內容 (Base64 圖片或文本)
        #print(associated_summary)
        if looks_like_base64(doc_content) and is_image_data(doc_content):
            # 如果是 Base64 圖片，解碼為 PIL.Image
            try:
                img_data = base64.b64decode(doc_content)
                pil_image = Image.open(io.BytesIO(img_data)).convert("RGB") # 確保是 RGB 格式
                context_images_pil.append(pil_image) # 將 PIL.Image 存入列表
                all_context_messages.append({"type": "image", "content": pil_image}) # 添加到消息列表
                all_context_messages.append({"type": "text", "text": associated_summary})#0924

                #all_context_messages.append({"type": "image", "text": }) # 添加到消息列表
                
            except Exception as e:
                print(f"轉換 Base64 圖片為 PIL.Image 失敗：{e}")
                all_context_messages.append({"type": "text", "text": f"[無法加載圖片: {doc_content[:50]}...]"})
        else:
            # 如果是文本，直接添加
            all_context_messages.append({"type": "text", "text": doc_content})
    combined_text_context = all_context_messages
    # 構建給 Llama 3.2 Vision 模型的最終文本提示
    rag_instruction_text = f"""提供的上下文文本和圖片資訊是口述影像描述的範例，請將問題修改成符合口述影像的版本。
    
上下文文本：
{combined_text_context}

問題：{user_question}
"""
    # 構建最終的 Llama 消息列表
    # Llama 3 Instruct 期望用戶的單個 'content' 字段是一個列表，可以包含多個圖片和文本。
    final_user_content_list = []

    # 先添加圖片內容（PIL.Image 物件）
    for img_pil in context_images_pil:
        final_user_content_list.append({"type": "image", "content": img_pil})

    # 然後添加文本指令和問題
    final_user_content_list.append({"type": "text", "text": rag_instruction_text})

    # 組合為最終的 messages 格式
    messages_for_llama = [{"role": "user", "content": final_user_content_list}]

    return context_images_pil, messages_for_llama


            
#-----------------------------------------------------------------------------------------
def give_Question():
    # 丟入使用者問題
    user_question = "畫面上方是藍天白雲，一位女子撐著一把綠色陽傘，穿著淺色長裙和淺色外套，裙襬蓬鬆，她站在畫面中央偏右，面向畫面左側。畫面下方是綠色的矮灌木叢，一位戴著淺色帽子的孩童站在畫面左側，面向畫面中央，穿著深色衣物。" 
    # 假設 retrieved_docs 已經包含了你檢索到的 Document 物件列表
    retrieved_docs = retriever_multi_vector_img.invoke(user_question)
    if retrieved_docs: # 確保列表不為空
        #print(f"retrieved_docs 的第一個元素型別是: {type(retrieved_docs[0])}")
        if isinstance(retrieved_docs[0], str):
            print(f"警告：檢索結果的元素是字串，而非 LangChain Document 物件。")
    
    llama_images_for_model, llama_messages = get_llama_vision_inputs_from_docs(user_question, retrieved_docs)#丟入使用者問題及檢索到的資訊
    
    #print(f"準備發送給 Llama 3.2 Vision 模型的圖片數量: {len(llama_images_for_model)} 張。")
    #print(f"準備發送給 Llama 3.2 Vision 模型的訊息結構: {llama_messages}") # 可以打印查看格式
    #如果你無法從提供的資訊中找到答案，請說明「根據提供的資訊無法回答此問題」。
    #不要引入你個人預設的知識。
    
    #6. 將結構化的訊息列表轉換為單一的文本字串，遵循 Llama 3 的對話格式
    # tokenize=False 讓 processor 在下一步進行分詞
    input_text_for_processor = processor.apply_chat_template(#將先前預先定義好的對話格式做轉換(長字串)
        llama_messages, 
        add_generation_prompt=True, # 添加模型的生成提示，讓模型知道接下來該生成答案(做標記在長字串後)
        tokenize=False # 不在這裡進行分詞，讓下一步的 processor() 統一處理
    )
    #print(f"經過 Chat Template 處理後的文本輸入預覽: {input_text_for_processor[:500]}...")
    
    #7. 將圖片 (PIL Image 列表) 和模板化後的文本字串一同傳遞給 processor
    # processor 會負責將它們轉換為 input_ids, attention_mask, pixel_values 等所有必要的張量
    inputs = processor(
        images=llama_images_for_model, # PIL Image 物件列表
        text=input_text_for_processor, # 經過 apply_chat_template 處理後的文本
        return_tensors="pt"            # 返回 PyTorch 張量
    ).to(model.device)                 # 將張量移動到模型所在的設備 (GPU/CPU)
    
    #print(f"準備發送給 Llama 3.2 Vision 模型的輸入張量 keys: {inputs.keys()}")
    #print(f"input_ids.shape: {inputs['input_ids'].shape}")
    if 'pixel_values' in inputs:
        print(f"pixel_values.shape: {inputs['pixel_values'].shape}")
    # 你可能會看到 'aspect_ratio_ids' 和 'aspect_ratio_mask' 等其他鍵，它們也在 inputs 字典中
    
    #8. 最後問答
    print("\n正在將整合後的輸入發送給 Llama 3.2 Vision 模型進行生成...")
    # --- 新增的程式碼：輸出完整提示詞 ---
    # 解碼 inputs["input_ids"] 來查看最終傳給模型的完整文本提示
    final_prompt_for_model = processor.decode(
        inputs["input_ids"][0], # inputs["input_ids"] 是一個批次，取第一個 (也是唯一一個) 序列
        skip_special_tokens=False # 這裡要保留特殊 token，因為它們是提示詞的一部分，幫助模型理解格式
    )
    print("\n--- 最終給 Llama 3.2 Vision 模型的完整提示詞 ---")
    print(final_prompt_for_model)
    print("--------------------------------------------------")
    # --- 新增程式碼結束 ---
    # 設置生成參數
    generate_kwargs = {
        "max_new_tokens": 1024, # 模型最多生成多少個新的 token
        "do_sample": True,       # 是否使用抽樣生成 (True 更有創造性，False 更確定性)
        "top_p": 0.9,            # top-p 抽樣的閾值
        "temperature": 0.7,      # 溫度，控制生成文本的隨機性 (0.0 最確定，1.0+ 更隨機)
        "pad_token_id": processor.tokenizer.eos_token_id, # 填充 token ID
        "eos_token_id": processor.tokenizer.eos_token_id,   # 結束 token ID，模型看到這個會停止生成
    }
    
    # --- 關鍵修改：只傳遞 'input_ids' 和 'attention_mask' ---
    # 這些是 generate 方法最通用的文本輸入參數。
    # 'pixel_values', 'aspect_ratio_ids', 'aspect_ratio_mask' 等可能已被編碼到 input_ids 中
    # 或由模型內部處理，不需要作為頂層 generate 參數。
    output = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        **generate_kwargs
    )
    print("\n--- Llama 3.2 Vision 生成的答案 ---")
    
    # 將生成的 token ID 解碼回文本
    # 這裡我們只解碼模型新生成的 token，忽略輸入提示的 token
    response_text = processor.decode(
        output[0][inputs["input_ids"].shape[-1]:], # 從輸入提示結束的位置開始解碼
        skip_special_tokens=True # 跳過特殊 token (例如 <pad>, <s>, </s>)
    )
    
    return response_text

ans = give_Question()#給問題---------------------------------------------------------------------------------------------------------------
print(f"最終{ans}")

def save_txt():
    save_txt_path = r"C:\Users\TYS\Desktop\RAG_UI\save_txt.txt" # 檔案的完整路徑和名稱
    
    try:
        # 使用 'w' 模式 (寫入)，如果檔案不存在會創建它，如果存在會覆蓋它。
        # 設置 encoding='utf-8' 以確保中文字元正確儲存。
        with open(save_txt_path, 'w', encoding='utf-8') as file:
            file.write(ans)
        
        print(f"成功將字串存入檔案：{save_txt_path}")
    
    except FileNotFoundError:
        print("錯誤：指定的路徑或目錄不存在。請檢查路徑是否正確。")
    except Exception as e:
        print(f"發生錯誤：{e}")
save_txt()

import asyncio
import nest_asyncio
import edge_tts
import pygame
import time
import os
def voice():
    # 修補 Spyder 中的事件迴圈
    nest_asyncio.apply()
    
    # 定義要使用的語音
    VOICE = "zh-TW-HsiaoChenNeural"  
    
    # 儲存檔案的路徑
    OUTPUT_FILE = r"C:\Users\TYS\Desktop\RAG_UI\ans.mp3"  
    
    # 嘗試刪除已存在的檔案
    if os.path.exists(OUTPUT_FILE):
        try:
            os.remove(OUTPUT_FILE)  # 刪除已存在的檔案
        except PermissionError:
            print(f"檔案 {OUTPUT_FILE} 目前正在被使用，稍後再刪除。")
    
    # 讀取文本檔案
    def read_text_from_file(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
            return text
        except FileNotFoundError:
            print(f"錯誤: 檔案 {file_path} 找不到！")
            return None
        except Exception as e:
            print(f"讀取檔案時發生錯誤: {e}")
            return None
    
    async def amain():
        # 設定檔案路徑
        input_file = r"C:\Users\TYS\Desktop\RAG_UI\save_txt.txt"   # 修改為你的文本檔案路徑
    
        # 從文本檔案讀取內容
        TEXT = read_text_from_file(input_file)
        
        if TEXT is None:
            print("未能成功讀取文本檔案。")
            return
    
        communicate = edge_tts.Communicate(TEXT, VOICE)
        try:
            await communicate.save(OUTPUT_FILE)
            print(f"語音已儲存為：{OUTPUT_FILE}")
        except PermissionError:
            print(f"無法儲存語音檔案 {OUTPUT_FILE}，因為檔案被鎖住或未授權。")
    
        print("正在播放語音...")
    
        # 初始化 pygame mixer
        pygame.mixer.init()
        pygame.mixer.music.load(OUTPUT_FILE)
        pygame.mixer.music.play()
    
        # 等待音樂播放完成
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
    
        # 播放完畢後釋放資源
        pygame.mixer.quit()
    
        # 嘗試刪除音頻檔案
        try:
            if os.path.exists(OUTPUT_FILE):
                os.remove(OUTPUT_FILE)
                print(f"檔案 {OUTPUT_FILE} 已成功刪除。")
        except PermissionError:
            print(f"無法刪除檔案 {OUTPUT_FILE}，檔案可能仍然被使用中。")
    
    loop = asyncio.get_event_loop()
    loop.run_until_complete(amain())

voice()
