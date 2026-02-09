
# 📚 RAG QA + TTS (سندمحور) — توضیح خط‌به‌خط نوت‌بوک
<div dir="rtl" style="text-align: right;">

این نوت‌بوک یک **RAG (Retrieval-Augmented Generation)** ساده می‌سازد:

**جریان کلی کار:**

1) سند را به چند «تکه/چانک» تقسیم می‌کند  
2) برای هر چانک embedding می‌سازد  
3) داخل FAISS ایندکس می‌کند  
4) برای سوال کاربر نزدیک‌ترین چانک‌ها را پیدا می‌کند  
5) با آن‌ها یک prompt می‌سازد  
6) با یک مدل سبک (Flan-T5) جواب تولید می‌کند  
7) جواب را با gTTS صوتی می‌کند  
8) همه را داخل یک رابط Gradio نشان می‌دهد  
</div>


## ✅ سلول 1 — نصب پکیج‌ها + بررسی Torch/GPU

```python
!pip -q install faiss-cpu sentence-transformers transformers accelerate gradio gTTS
````

**این سلول پکیج‌های لازم را نصب می‌کند:**
<div dir="rtl" style="text-align: right;">
* `faiss-cpu`: جستجوی برداری سریع (Vector Search) روی CPU
* `sentence-transformers`: ساخت embedding برای متن
* `transformers`, `accelerate`: اجرای مدل‌های HuggingFace
* `gradio`: ساخت UI وب برای چت
* `gTTS`: تبدیل متن به صدا با Google Text-to-Speech

> `-q` یعنی خروجی نصب کم‌حرف‌تر باشد.
</div>


```python
import torch
print('torch:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
```
<div dir="rtl" style="text-align: right;">
* `torch` را وارد می‌کند
* نسخه‌ی Torch را چاپ می‌کند
* بررسی می‌کند آیا CUDA (GPU) در دسترس است یا نه
</div>



## ✅ سلول 2 — ایمپورت‌ها و انتخاب دستگاه (CPU/GPU)

```python
import os
import re
import numpy as np
import faiss
from dataclasses import dataclass
```
<div dir="rtl" style="text-align: right;">
* `os`: کار با مسیرها/فایل‌ها (اینجا خیلی استفاده نشده)
* `re`: regex برای پاکسازی متن
* `numpy`: آرایه‌های عددی (embeddingها)
* `faiss`: ساخت index و جستجوی شباهت
* `dataclass`: ایمپورت شده ولی استفاده نشده (می‌تواند حذف شود)
</div>


```python
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
```
<div dir="rtl" style="text-align: right;">
* `SentenceTransformer`: مدل embedding
* `AutoTokenizer` و `AutoModelForSeq2SeqLM`: توکنایزر و مدل تولید متن (seq2seq)
* دوباره `torch` ایمپورت شده (تکراری است ولی مشکلی ایجاد نمی‌کند)
</div>
```python
from gtts import gTTS
import gradio as gr
```
<div dir="rtl" style="text-align: right;">
* `gTTS`: متن → mp3
* `gradio`: UI
</div>


```python
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print('DEVICE:', DEVICE)
```

* اگر GPU باشد روی `cuda` می‌رود وگرنه `cpu`
* دستگاه انتخابی را چاپ می‌کند



## ✅ سلول 3 — متن سند + تابع خواندن فایل txt

```python
DOCUMENT_TEXT = """
""".strip()
```
<div dir="rtl" style="text-align: right;">

* متغیر اصلی سند
* فعلاً خالی است
* `.strip()` فاصله‌های ابتدا/انتها را حذف می‌کند
</div>


```python
def load_txt_from_path(path: str) -> str:
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()
```
<div dir="rtl" style="text-align: right;">
* تابع ساده برای خواندن فایل متنی UTF-8
* `with open(...)` یعنی فایل بعد از خواندن خودکار بسته می‌شود
</div>


```python
print('Document chars:', len(DOCUMENT_TEXT))
```
<div dir="rtl" style="text-align: right;">
* تعداد کاراکترهای سند فعلی را چاپ می‌کند (اینجا احتمالاً 0)
</div>


## ✅ سلول 4 — نرمال‌سازی متن + چانک‌کردن

### 1) نرمال‌سازی متن

```python
def normalize_text(text: str) -> str:
    text = text.replace('\u200c', ' ')  # ZWNJ
    text = re.sub(r"\s+", " ", text).strip()
    return text
```
<div dir="rtl" style="text-align: right;">
* `\u200c` همان **نیم‌فاصله (ZWNJ)** است؛ آن را به فاصله تبدیل می‌کند تا متن یکنواخت شود
* `re.sub(r"\s+", " ", text)`: هر تعداد whitespace (فاصله/خط جدید/تب) → یک فاصله
* `strip()`: حذف فاصله‌های ابتدا و انتها
</div>


### 2) چانک‌کردن متن

```python
def chunk_text(text: str, chunk_size: int = 450, overlap: int = 80):
    """Chunking ساده بر اساس تعداد کاراکتر
    chunk_size و overlap قابل تغییر هستند
    """
```
<div dir="rtl" style="text-align: right;">
* تکه‌تکه کردن متن بر اساس **تعداد کاراکتر**
* `chunk_size`: طول هر تکه
* `overlap`: همپوشانی بین تکه‌ها (برای اینکه مطالب مرزی از دست نروند)
</div>


```python
    text = normalize_text(text)
    if not text:
        return []
```
<div dir="rtl" style="text-align: right;">
* ابتدا متن را نرمال می‌کند
* اگر خالی بود لیست خالی برمی‌گرداند
</div>


```python
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
```
<div dir="rtl" style="text-align: right;">
* از `start` تا `end` یک برش می‌گیرد
* `min` برای اینکه از طول متن جلوتر نرود
* اگر تکه خالی نبود به لیست اضافه می‌کند
</div>


```python
        start = end - overlap
        if start < 0:
            start = 0
        if end == len(text):
            break
```
<div dir="rtl" style="text-align: right;">
* برای چانک بعدی، `start` را عقب‌تر می‌برد تا overlap ایجاد شود
* اگر منفی شد صفر می‌کند
* اگر به انتهای متن رسیدیم، حلقه تمام می‌شود
</div>

```python
    return chunks
```

* خروجی: لیست چانک‌ها



```python
chunks = chunk_text(DOCUMENT_TEXT, chunk_size=450, overlap=80)
print('Num chunks:', len(chunks))
print('Sample chunk:\n', chunks[0][:300] if chunks else 'EMPTY')
```
<div dir="rtl" style="text-align: right;">
* چانک‌های سند فعلی را می‌سازد
* تعدادشان را چاپ می‌کند
* اگر چانک وجود داشت ۳۰۰ کاراکتر اول چانک اول را نشان می‌دهد
</div>

## ✅ سلول 5 — embedding گرفتن از چانک‌ها

```python
EMBED_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
embedder = SentenceTransformer(EMBED_MODEL_NAME)
```
<div dir="rtl" style="text-align: right;">
* اسم مدل embedding چندزبانه (برای فارسی هم مناسب)
* مدل را لود می‌کند
</div>


```python
def embed_texts(texts):
    vecs = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=True, normalize_embeddings=True)
    return vecs.astype('float32')
```
<div dir="rtl" style="text-align: right;">
* `encode`: متن‌ها را به بردار تبدیل می‌کند
* `convert_to_numpy=True`: خروجی numpy array
* `show_progress_bar=True`: نمایش نوار پیشرفت
* `normalize_embeddings=True`: نرمال‌سازی بردارها (برای شباهت بهتر)
* `astype('float32')`: FAISS معمولاً float32 دوست دارد
</div>


```python
chunk_embeddings = embed_texts(chunks) if chunks else np.zeros((0, 384), dtype='float32')
print('Embeddings shape:', chunk_embeddings.shape)
```
<div dir="rtl" style="text-align: right;">

* اگر چانک داریم embedding می‌گیرد
* اگر نداریم آرایه خالی با شکل `(0, 384)` می‌سازد (۳۸۴ ابعاد این مدل است)
* شکل embeddingها را چاپ می‌کند
</div>


## ✅ سلول 6 — ساخت ایندکس FAISS

```python
def build_faiss_index(embeddings: np.ndarray):
    if embeddings.size == 0:
        return None
```

* اگر embedding خالی باشد، index ساخته نمی‌شود



```python
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index
```
<div dir="rtl" style="text-align: right;">
* `dim`: تعداد ابعاد بردار
* `IndexFlatIP`: ایندکس ساده با معیار **Inner Product**
  چون embeddingها نرمال شده‌اند، `inner product ≈ cosine similarity`
* `add`: همه embeddingها را وارد ایندکس می‌کند
</div>


```python
index = build_faiss_index(chunk_embeddings)
print('FAISS index ready:', index is not None)
```

* ایندکس ساخته می‌شود و آماده بودنش چاپ می‌شود



## ✅ سلول 7 — بازیابی Top-k چانک‌های مرتبط

```python
def retrieve_top_k(query: str, k: int = 4):
    if index is None or not chunks:
        return []
```

* اگر ایندکس/چانک نداریم، خروجی خالی



```python
    q_emb = embed_texts([query])
    scores, ids = index.search(q_emb, k)
```
<div dir="rtl" style="text-align: right;">
* embedding سوال را می‌سازد (یک سوال → یک بردار)
* `search`: نزدیک‌ترین `k` بردار را می‌دهد

  * `ids`: اندیس چانک‌ها
  * `scores`: امتیاز شباهت
</div>


```python
    ids = ids[0].tolist()
    scores = scores[0].tolist()
```
<div dir="rtl" style="text-align: right;">
* چون خروجی دوبعدی است (batch)، سطر اول را می‌گیرد و به لیست تبدیل می‌کند
</div>


```python
    results = []
    for i, s in zip(ids, scores):
        if i == -1:
            continue
        results.append((chunks[i], float(s), i))
    return results
```
<div dir="rtl" style="text-align: right;">
* اگر `-1` بود نتیجه نامعتبر است
* خروجی هر نتیجه: `(متن چانک، امتیاز، اندیس)`
</div>


```python
test_q = "موضوع سند چیست؟"
print(retrieve_top_k(test_q, k=3)[:1])
```
<div dir="rtl" style="text-align: right;">
* تست: سوال می‌پرسد و ۱ نتیجه اول را چاپ می‌کند
</div>

<div dir="rtl" style="text-align: right;">
## ✅ سلول 8 — ساخت prompt برای مدل زبانی

</div>

```python
def build_prompt(context_chunks, question: str) -> str:
    context = "\n\n".join([f"[{i}] {c}" for i, c in enumerate(context_chunks, start=1)])
```

* چانک‌های انتخاب‌شده را شماره‌گذاری و یکپارچه می‌کند



```python
    prompt = f"""
You are a QA assistant.

RULES:
1) Answer ONLY using the provided CONTEXT.
2) If the answer is not in the context, say exactly: "اطلاعات کافی در متن موجود نیست."
3) Keep the answer concise and well-structured.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER (in Persian):
""".strip()
```
<div dir="rtl" style="text-align: right;">
* prompt انگلیسی است ولی می‌گوید جواب **فارسی** باشد
* قانون مهم: فقط از `CONTEXT` استفاده کن؛ اگر نبود دقیقاً همان جمله را بگو

</div>


```python
    return prompt
```
<div dir="rtl" style="text-align: right;">
* prompt نهایی برمی‌گردد
</div>


## ✅ سلول 9 — لود مدل تولید پاسخ + تولید پاسخ

```python
LLM_NAME = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(LLM_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(LLM_NAME)
model.to(DEVICE)
```
<div dir="rtl" style="text-align: right;">
* مدل سبک `flan-t5-small` را لود می‌کند
* توکنایزر و مدل را می‌سازد
* روی CPU یا GPU می‌برد
</div>


```python
def generate_answer(prompt: str, max_new_tokens: int = 180):
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=1024).to(DEVICE)
```
<div dir="rtl" style="text-align: right;">
* prompt را توکنایز می‌کند
* `truncation=True`: اگر طولانی شد قطع کند
* `max_length=1024`: سقف طول ورودی
* داده‌ها را روی `DEVICE` می‌برد
</div>


```python
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=4,
            do_sample=False,
        )
```
<div dir="rtl" style="text-align: right;">
* `no_grad`: inference بدون محاسبه گرادیان
* `generate`:

  * `max_new_tokens`: حداکثر طول جواب
  * `num_beams=4`: beam search برای جواب بهتر
  * `do_sample=False`: تصادفی نیست (پایدارتر)
</div>


```python
    text = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    return text
```

* تبدیل خروجی توکن‌ها به متن و برگشت دادن جواب



### تست سریع

```python
if chunks:
    r = retrieve_top_k("سند درباره چیست؟", k=4)
    ctx = [x[0] for x in r]
    p = build_prompt(ctx, "سند درباره چیست؟")
    print(generate_answer(p))
else:
    print('Document is empty. Paste or upload a .txt first.')
```
<div dir="rtl" style="text-align: right;">

* اگر سند داریم: retrieval → prompt → جواب چاپ می‌شود
* اگر نداریم: پیام سند خالی است
</div>


## ✅ سلول 10 — متن به صدا (TTS)

```python
def text_to_speech(text, out_path="answer.mp3", lang="en"):
    text = (text or "").strip()
    if not text:
        return None
```
<div dir="rtl" style="text-align: right;">
* متن را امن می‌کند (اگر `None` بود، رشته خالی)
* اگر خالی بود `None` برمی‌گرداند
</div>


```python
    gTTS(text=text, lang=lang).save(out_path)
    return out_path
```
<div dir="rtl" style="text-align: right;">
* با gTTS فایل mp3 می‌سازد و مسیرش را برمی‌گرداند
</div>


## ✅ سلول 11— تابع اصلی RAG

```python
def rag_answer(question: str, top_k: int = 4):
    if not DOCUMENT_TEXT.strip() or not chunks:
        return "ابتدا متن سند را وارد کنید (Paste یا فایل .txt).", []
```
<div dir="rtl" style="text-align: right;">
* اگر سند/چانک نداریم: پیام خطا + لیست خالی
</div>


```python
    retrieved = retrieve_top_k(question, k=top_k)
    context_chunks = [c for (c, s, idx) in retrieved]
    prompt = build_prompt(context_chunks, question)
    answer = generate_answer(prompt)
    return answer, retrieved
```
<div dir="rtl" style="text-align: right;">

* retrieval انجام می‌دهد
* فقط متن چانک‌ها را جدا می‌کند
* prompt می‌سازد
* جواب تولید می‌کند
* خروجی: `(answer, retrieved_details)`
</div>


## ✅ سلول 12 — بازسازی pipeline + رابط Gradio

### 1) بازسازی ایندکس با سند جدید

```python
def rebuild_pipeline_with_new_doc(doc_text: str, chunk_size: int = 450, overlap: int = 80):
    global DOCUMENT_TEXT, chunks, chunk_embeddings, index
```

* چون می‌خواهد متغیرهای سراسری را تغییر دهد، `global` می‌گذارد



```python
    DOCUMENT_TEXT = (doc_text or "").strip()
    chunks = chunk_text(DOCUMENT_TEXT, chunk_size=chunk_size, overlap=overlap)
```

* متن سند را ذخیره می‌کند
* چانک می‌کند



```python
    if chunks:
        chunk_embeddings = embed_texts(chunks)
        index = build_faiss_index(chunk_embeddings)
    else:
        chunk_embeddings = np.zeros((0, 384), dtype='float32')
        index = None
    return f"✅ سند بارگذاری شد. تعداد chunk: {len(chunks)}"
```
<div dir="rtl" style="text-align: right;">
* اگر چانک هست: embedding → index
* اگر نیست: همه چیز خالی
* پیام وضعیت برمی‌گرداند
</div>


### 2) خواندن فایل آپلودی Gradio

```python
def read_uploaded_file(file_obj):
    if file_obj is None:
        return ""
```
<div dir="rtl" style="text-align: right;">
سپس نوع‌های مختلفی که Gradio ممکن است برگرداند را پوشش می‌دهد:
</div>

```python
    if isinstance(file_obj, str):
        path = file_obj
    elif hasattr(file_obj, "name"):
        path = file_obj.name
    elif isinstance(file_obj, dict) and "path" in file_obj:
        path = file_obj["path"]
    elif hasattr(file_obj, "path"):
        path = file_obj.path
    else:
        raise TypeError(...)
```

بعد فایل را می‌خواند:

```python
    with open(path, "r", encoding="utf-8") as f:
        return f.read()
```


<div dir="rtl" style="text-align: right;">
### 3) تابع چت (ورودی کاربر → جواب + سورس‌ها + صوت)
</div>

```python
def chat_fn(message, history, top_k, chunk_size, overlap):
    if not (DOCUMENT_TEXT and DOCUMENT_TEXT.strip()):
        return (history or []), "❌ ابتدا سند را وارد کنید.", None
```
<div dir="rtl" style="text-align: right;">
* اگر سند نیست، ۳ خروجی می‌دهد چون UI سه خروجی دارد:

  1. history چت
  2. markdown سورس‌ها
  3. audio (هیچی)

</div>


```python
    answer, retrieved = rag_answer(message, top_k=int(top_k))
    answer = (answer or "").strip()
```
* جواب RAG
* تمیزکاری



```python
    if not answer:
        answer = "متأسفانه پاسخ قابل تولید نیست..."
```

* اگر خالی بود پیام fallback می‌دهد



```python
    audio_path = text_to_speech(answer, out_path="answer.mp3", lang="en")
```

* جواب را صوتی می‌کند

> بهتر: `lang="fa"` اگر فارسی است



```python
    sources_md = "\n\n".join([
        f"**Chunk #{idx} | score={score:.3f}**\n\n{chunk[:700]}"
        for (chunk, score, idx) in retrieved
    ])
```

* چانک‌های بازیابی‌شده را به markdown تبدیل می‌کند (تا ۷۰۰ کاراکتر از هر چانک)



```python
    history = (history or []) + [(message, answer)]
    return history, sources_md, audio_path
```

* history را آپدیت می‌کند و خروجی‌ها را برمی‌گرداند



### 4) ساخت UI با Gradio

```python
with gr.Blocks() as demo:
    gr.Markdown("# RAG QA + TTS (سندمحور)")
```

* یک اپ Gradio می‌سازد

**بخش ورودی سند:**

* `doc_paste = gr.Textbox(...)`
* `doc_file = gr.File(...)`
* `chunk_size = gr.Slider(...)`
* `overlap = gr.Slider(...)`
* `load_btn = gr.Button(...)`
* `load_status = gr.Textbox(...)`

**بخش چت:**

* `chatbot = gr.Chatbot(...)`
* `msg = gr.Textbox(...)`
* `top_k = gr.Slider(...)`
* `sources = gr.Markdown(...)`
* `audio = gr.Audio(..., type="filepath")`


### 5) تابع `on_load` (وقتی paste/file/slider تغییر کند)

```python
def on_load(doc_text, file_obj, chunk_size, overlap):
  if doc_text and doc_text.strip():
      chosen_text = doc_text
  elif file_obj is not None:
      chosen_text = read_uploaded_file(file_obj)
  else:
      chosen_text = ""
```
<div dir="rtl" style="text-align: right;">
* اولویت با paste است؛ اگر خالی بود از فایل می‌خواند
</div>


```python
  if not (chosen_text and chosen_text.strip()):
      return "❌ سند خالی است..."
  return rebuild_pipeline_with_new_doc(chosen_text, int(chunk_size), int(overlap))
```

* اگر سند خالی بود پیام می‌دهد
* وگرنه ایندکس را می‌سازد



### 6) اتصال رویدادها (Event Handlers)

```python
load_btn.click(on_load, inputs=[...], outputs=[load_status])
doc_paste.change(on_load, ...)
doc_file.change(on_load, ...)
chunk_size.change(on_load, ...)
overlap.change(on_load, ...)
```

* با هر تغییر، ایندکس دوباره ساخته می‌شود


**ارسال پیام چت:**

```python
msg.submit(chat_fn, inputs=[msg, chatbot, top_k, chunk_size, overlap], outputs=[chatbot, sources, audio])
msg.submit(lambda: "", None, msg)
```
<div dir="rtl" style="text-align: right;">
* submit اول: جواب را می‌گیرد
* submit دوم: textbox پیام را خالی می‌کند
</div>


### اجرای اپ

```python
demo.launch(share=True, debug=True)
```

* اپ را اجرا می‌کند
* `share=True`: لینک عمومی موقت می‌دهد
* `debug=True`: خطاها را دقیق‌تر چاپ می‌کند
