# import os
# os.environ["TRANSFORMERS_NO_TF"] = "1"
# os.environ["USE_TF"] = "0"
# import torch, streamlit as st
# from dotenv import load_dotenv
# from PyPDF2 import PdfReader
# from langchain.text_splitter import CharacterTextSplitter
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain.memory import ConversationBufferMemory
# from langchain.chains import ConversationalRetrievalChain
# from htmlTemplates import css, bot_template, user_template
# from langchain_community.llms import HuggingFaceHub

# def get_pdf_text(pdf_docs):
#     text = ""
#     for pdf in pdf_docs:
#         pdf_reader = PdfReader(pdf)
#         for page in pdf_reader.pages:
#             text += page.extract_text()
#     return text


# def get_text_chunks(text):
#     text_splitter = CharacterTextSplitter(
#         separator="\n",
#         chunk_size=1000,
#         chunk_overlap=200,
#         length_function=len
#     )
#     chunks = text_splitter.split_text(text)
#     return chunks


# def get_vectorstore(text_chunks):
#     #embeddings = OpenAIEmbeddings()
#     embeddings = HuggingFaceEmbeddings(model_name="hkunlp/instructor-xl")
#     vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
#     return vectorstore


# def get_conversation_chain(vectorstore):
#     #llm = ChatOpenAI()
#     llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
#     llm = HuggingFaceHub(
#     repo_id="google/flan-t5-xl",           
#     task="text2text-generation",           
#     huggingfacehub_api_token=os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN"),
#     model_kwargs={"temperature": 0.5, "max_new_tokens": 512}
# )

#     memory = ConversationBufferMemory(
#         memory_key='chat_history', return_messages=True)
#     conversation_chain = ConversationalRetrievalChain.from_llm(
#         llm=llm,
#         retriever=vectorstore.as_retriever(),
#         memory=memory
#     )
#     return conversation_chain


# def handle_userinput(user_question):
#     response = st.session_state.conversation({'question': user_question})
#     st.session_state.chat_history = response['chat_history']

#     for i, message in enumerate(st.session_state.chat_history):
#         if i % 2 == 0:
#             st.write(user_template.replace(
#                 "{{MSG}}", message.content), unsafe_allow_html=True)
#         else:
#             st.write(bot_template.replace(
#                 "{{MSG}}", message.content), unsafe_allow_html=True)


# def main():
#     load_dotenv()
#     st.set_page_config(page_title="Chat with multiple PDFs",
#                        page_icon=":books:")
#     st.write(css, unsafe_allow_html=True)

#     if "conversation" not in st.session_state:
#         st.session_state.conversation = None
#     if "chat_history" not in st.session_state:
#         st.session_state.chat_history = None

#     st.header("Chat with multiple PDFs :books:")
#     user_question = st.text_input("Ask a question about your documents:")
#     if user_question:
#         handle_userinput(user_question)

#     with st.sidebar:
#         st.subheader("Your documents")
#         pdf_docs = st.file_uploader(
#             "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
#         if st.button("Process"):
#             with st.spinner("Processing"):
#                 # get pdf text
#                 raw_text = get_pdf_text(pdf_docs)

#                 # get the text chunks
#                 text_chunks = get_text_chunks(raw_text)

#                 # create vector store
#                 vectorstore = get_vectorstore(text_chunks)

#                 # create conversation chain
#                 st.session_state.conversation = get_conversation_chain(
#                     vectorstore)


# if __name__ == '__main__':
#     main()
# ==== keep TensorFlow/Keras out of Transformers ====
# ==== keep TensorFlow/Keras out of Transformers ====
# ==== keep TensorFlow/Keras out of Transformers (must be first) ====
# ==== keep TensorFlow/Keras out of Transformers (must be first) ====
# ==== keep TensorFlow/Keras out of Transformers (must be first) ====
# ==== keep TensorFlow/Keras out of Transformers (must be first) ====
# ==== keep TensorFlow/Keras out of Transformers (must be first) ====
# ==== keep TensorFlow/Keras out of Transformers (must be first) ====
# ==== keep TensorFlow/Keras out of Transformers (must be first) ====
# ==== keep TensorFlow/Keras out of Transformers (must be first) ====

########################################################################

# import os
# os.environ["TRANSFORMERS_NO_TF"] = "1"
# os.environ["USE_TF"] = "0"
# os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # quiet MPS quirks on macOS

# import torch
# import streamlit as st
# from dotenv import load_dotenv
# from PyPDF2 import PdfReader

# # LangChain bits (modern)
# from langchain.text_splitter import CharacterTextSplitter
# from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
# from langchain_community.vectorstores import FAISS
# from langchain.memory import ConversationBufferMemory
# from langchain.chains import ConversationalRetrievalChain



# # your HTML templates
# from htmlTemplates import css, bot_template, user_template


# # -----------------------------
# # Helpers
# # -----------------------------
# def get_pdf_text(pdf_docs):
#     text = ""
#     for pdf in (pdf_docs or []):
#         pdf_reader = PdfReader(pdf)
#         for page in pdf_reader.pages:
#             text += page.extract_text() or ""   # coalesce None
#     return text


# def get_text_chunks(text: str):
#     text_splitter = CharacterTextSplitter(
#         separator="\n",
#         chunk_size=1000,
#         chunk_overlap=200,
#         length_function=len,
#     )
#     return text_splitter.split_text(text)


# @st.cache_resource
# def load_embedder():
#     """Cache the embedder so Streamlit hot-reloads don't re-download the model."""
#     device = "mps" if torch.backends.mps.is_available() else (
#         "cuda" if torch.cuda.is_available() else "cpu"
#     )
#     # Non-Instructor, super fast & reliable
#     return HuggingFaceEmbeddings(
#         model_name="sentence-transformers/all-MiniLM-L6-v2",
#         model_kwargs={"device": device},
#         encode_kwargs={"normalize_embeddings": True},
#     )


# def get_vectorstore(text_chunks):
#     embeddings = load_embedder()
#     return FAISS.from_texts(texts=text_chunks, embedding=embeddings)



# def build_llm(token: str):
#     # Zephyr prefers the conversational task with the provider you hit earlier.
#     return HuggingFaceEndpoint(
#         repo_id="HuggingFaceH4/zephyr-7b-beta",
#         task="conversational",              # <-- IMPORTANT
#         huggingfacehub_api_token=token,
#         temperature=0.1,
#         max_new_tokens=256,
#     )



# def get_conversation_chain(vectorstore):
#     # Prefer the API token name most folks use; fall back to the alternative env var
#     token = (
#         os.getenv("HUGGINGFACEHUB_API_TOKEN")
#         or os.getenv("HUGGINGFACE_HUB_TOKEN")
#     )
#     if not token:
#         st.error("Missing Hugging Face token. Add HUGGINGFACEHUB_API_TOKEN to your .env.")
#         st.stop()

#     llm = build_llm(token)

#     memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

#     return ConversationalRetrievalChain.from_llm(
#         llm=llm,
#         retriever=vectorstore.as_retriever(),
#         memory=memory,
#     )


# def handle_userinput(user_question):
#     response = st.session_state.conversation({"question": user_question})
#     st.session_state.chat_history = response["chat_history"]

#     for i, message in enumerate(st.session_state.chat_history):
#         if i % 2 == 0:
#             st.write(
#                 user_template.replace("{{MSG}}", message.content),
#                 unsafe_allow_html=True,
#             )
#         else:
#             st.write(
#                 bot_template.replace("{{MSG}}", message.content),
#                 unsafe_allow_html=True,
#             )


# # -----------------------------
# # Streamlit app
# # -----------------------------
# def main():
#     load_dotenv()  # loads .env so the HF token is available
#     st.set_page_config(page_title="Chat with multiple PDFs", page_icon="📚")
#     st.write(css, unsafe_allow_html=True)

#     # Quick reset to avoid stale chains after code changes
#     with st.sidebar:
#         if st.button("Reset app"):
#             st.session_state.clear()
#             st.experimental_rerun()

#     if "conversation" not in st.session_state:
#         st.session_state.conversation = None
#     if "chat_history" not in st.session_state:
#         st.session_state.chat_history = None

#     st.header("Chat with multiple PDFs 📚")
#     user_question = st.text_input("Ask a question about your documents:")
#     if user_question and st.session_state.conversation:
#         handle_userinput(user_question)

#     with st.sidebar:
#         st.subheader("Your documents")
#         pdf_docs = st.file_uploader(
#             "Upload your PDFs here and click on 'Process'",
#             accept_multiple_files=True,
#             type=["pdf"],
#         )
#         if st.button("Process", disabled=not pdf_docs):
#             with st.spinner("Processing..."):
#                 # 1) read PDFs
#                 raw_text = get_pdf_text(pdf_docs)
#                 if not raw_text.strip():
#                     st.warning("No text found in the uploaded PDFs.")
#                     st.stop()

#                 # 2) split into chunks
#                 text_chunks = get_text_chunks(raw_text)

#                 # 3) build vector store
#                 vectorstore = get_vectorstore(text_chunks)

#                 # 4) build conversation chain
#                 st.session_state.conversation = get_conversation_chain(vectorstore)

#                 st.success("Ready! Ask a question above.")

#     if not st.session_state.conversation:
#         st.info("Upload PDFs and click **Process** to start.")


# if __name__ == "__main__":
#     main()



###############################################################
# ==== keep TensorFlow/Keras out of Transformers (must be first) ====
# ==== keep TensorFlow/Keras out of Transformers (must be first) ====
# ==== keep TensorFlow/Keras out of Transformers (must be first) ====
# import os
# os.environ["TRANSFORMERS_NO_TF"] = "1"
# os.environ["USE_TF"] = "0"
# os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # macOS MPS quirks

# import torch
# import streamlit as st
# from dotenv import load_dotenv
# from PyPDF2 import PdfReader

# # LangChain (modern)
# from langchain.text_splitter import CharacterTextSplitter
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain.memory import ConversationBufferMemory
# from langchain.chains import ConversationalRetrievalChain

# # Local transformers pipeline
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
# from langchain_community.llms import HuggingFacePipeline

# # Your HTML templates
# from htmlTemplates import css, bot_template, user_template


# # -----------------------------
# # Helpers
# # -----------------------------
# def get_pdf_text(pdf_docs):
#     text = ""
#     for pdf in (pdf_docs or []):
#         pdf_reader = PdfReader(pdf)
#         for page in pdf_reader.pages:
#             text += page.extract_text() or ""   # coalesce None
#     return text


# def get_text_chunks(text: str):
#     splitter = CharacterTextSplitter(
#         separator="\n",
#         chunk_size=1000,
#         chunk_overlap=200,
#         length_function=len,
#     )
#     return splitter.split_text(text)


# @st.cache_resource
# def load_embedder():
#     """Cache the embedder so Streamlit hot-reloads don't re-download the model."""
#     device = "mps" if torch.backends.mps.is_available() else (
#         "cuda" if torch.cuda.is_available() else "cpu"
#     )
#     return HuggingFaceEmbeddings(
#         model_name="sentence-transformers/all-MiniLM-L6-v2",
#         model_kwargs={"device": device},
#         encode_kwargs={"normalize_embeddings": True},
#     )


# @st.cache_resource
# def load_local_llm():
#     """
#     Load FLAN-T5-base locally (good pick for an 8GB Mac).
#     If you see speed/memory issues, you can still lower to `google/flan-t5-small`.
#     """
#     model_id = "google/flan-t5-base"

#     # Pick device + dtype (float16 on GPU/MPS, float32 on CPU)
#     if torch.backends.mps.is_available():
#         device = "mps"
#         dtype = torch.float16
#     elif torch.cuda.is_available():
#         device = "cuda"
#         dtype = torch.float16
#     else:
#         device = "cpu"
#         dtype = torch.float32

#     tok = AutoTokenizer.from_pretrained(model_id)

#     model = AutoModelForSeq2SeqLM.from_pretrained(
#         model_id,
#         torch_dtype=dtype if device != "cpu" else torch.float32,
#         low_cpu_mem_usage=True,
#     )
#     if device != "cpu":
#         model.to(device)

#     gen = pipeline(
#         task="text2text-generation",
#         model=model,
#         tokenizer=tok,
#         max_new_tokens=256,
#         do_sample=False,          # deterministic for Q&A
#         # (no device=int; we already moved model to the right device)
#     )
#     return HuggingFacePipeline(pipeline=gen)


# def get_vectorstore(text_chunks):
#     embeddings = load_embedder()
#     return FAISS.from_texts(texts=text_chunks, embedding=embeddings)


# def get_conversation_chain(vectorstore):
#     llm = load_local_llm()  # local pipeline; no HF Inference API needed
#     st.caption("LLM ready: local google/flan-t5-base (text2text-generation)")
#     memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

#     return ConversationalRetrievalChain.from_llm(
#         llm=llm,
#         retriever=vectorstore.as_retriever(),
#         memory=memory,
#     )


# def handle_userinput(user_question):
#     response = st.session_state.conversation({"question": user_question})
#     st.session_state.chat_history = response["chat_history"]

#     for i, message in enumerate(st.session_state.chat_history):
#         if i % 2 == 0:
#             st.write(
#                 user_template.replace("{{MSG}}", message.content),
#                 unsafe_allow_html=True,
#             )
#         else:
#             st.write(
#                 bot_template.replace("{{MSG}}", message.content),
#                 unsafe_allow_html=True,
#             )


# # -----------------------------
# # Streamlit app
# # -----------------------------
# def main():
#     load_dotenv()
#     st.set_page_config(page_title="Chat with multiple PDFs", page_icon="📚")
#     st.write(css, unsafe_allow_html=True)

#     # Reset to avoid stale chains after code/config changes
#     with st.sidebar:
#         if st.button("Reset app"):
#             st.session_state.clear()
#             st.experimental_rerun()

#     if "conversation" not in st.session_state:
#         st.session_state.conversation = None
#     if "chat_history" not in st.session_state:
#         st.session_state.chat_history = None

#     st.header("Chat with multiple PDFs 📚")
#     user_question = st.text_input("Ask a question about your documents:")
#     if user_question and st.session_state.conversation:
#         handle_userinput(user_question)

#     with st.sidebar:
#         st.subheader("Your documents")
#         pdf_docs = st.file_uploader(
#             "Upload your PDFs here and click on 'Process'",
#             accept_multiple_files=True,
#             type=["pdf"],
#         )
#         if st.button("Process", disabled=not pdf_docs):
#             with st.spinner("Processing..."):
#                 raw_text = get_pdf_text(pdf_docs)
#                 if not raw_text.strip():
#                     st.warning("No text found in the uploaded PDFs.")
#                     st.stop()

#                 text_chunks = get_text_chunks(raw_text)
#                 vectorstore = get_vectorstore(text_chunks)
#                 st.session_state.conversation = get_conversation_chain(vectorstore)

#                 st.success("Ready! Ask a question above.")

#     if not st.session_state.conversation:
#         st.info("Upload PDFs and click **Process** to start.")


# if __name__ == "__main__":
#     main()
################################working model with google/flan-t5-xl but very bad

# ==== keep TensorFlow/Keras out of Transformers (must be first) ====
# ==== keep TensorFlow/Keras out of Transformers (must be first) ====
# ==== keep TensorFlow/Keras out of Transformers (must be first) ====
# ==== keep TensorFlow/Keras out of Transformers (must be first) ====
import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["USE_TF"] = "0"
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # macOS MPS quirks

import torch
import streamlit as st
from io import BytesIO
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from pdfminer.high_level import extract_text as pdfminer_extract_text

# LangChain (modern)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain

# Local transformers pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# Your HTML templates
from htmlTemplates import css, bot_template, user_template


# -----------------------------
# Robust PDF text extraction
# -----------------------------
def get_pdf_text(pdf_docs):
    """
    Robust text extraction:
      1) Try PyPDF2 for embedded text
      2) Fallback to pdfminer.six (handles more layouts)
      3) If still empty, flag as OCR candidate (commented OCR block included)
    """
    total_text = []
    ocr_candidates = 0

    for up in (pdf_docs or []):
        data = up.read()
        if not data:
            continue

        # --- Try PyPDF2 ---
        chunk = ""
        try:
            reader = PdfReader(BytesIO(data))
            for page in reader.pages:
                chunk += page.extract_text() or ""
        except Exception:
            pass

        # --- Fallback: pdfminer.six ---
        if len(chunk.strip()) < 30:
            try:
                chunk = pdfminer_extract_text(BytesIO(data)) or ""
            except Exception:
                pass

        # --- Mark for OCR if still empty ---
        if len(chunk.strip()) < 30:
            ocr_candidates += 1

            # OPTIONAL OCR (requires Tesseract & pypdfium2)
            # import pypdfium2, pytesseract
            # images = []
            # pdf = pypdfium2.PdfDocument(BytesIO(data))
            # for i in range(len(pdf)):
            #     page = pdf.get_page(i)
            #     bmp = page.render(scale=2).to_pil()
            #     images.append(bmp)
            # page_text = [pytesseract.image_to_string(img) for img in images]
            # chunk = "\n".join(page_text)

        total_text.append(chunk)

    text = "\n".join(total_text)
    if not text.strip():
        st.error(
            "No extractable text found. This PDF is likely scanned or image-based. "
            "Enable OCR in the code (comments) or run the file through an OCR tool like `ocrmypdf`."
        )
    elif ocr_candidates:
        st.warning(
            f"{ocr_candidates} file(s) had little/no embedded text. If answers look empty, enable OCR in the code or pre-OCR your PDFs."
        )

    return text


# -----------------------------
# Chunking (fit T5's context)
# -----------------------------
def get_text_chunks(text: str):
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=500,     # small to fit T5's ~512-token window w/ prompt
        chunk_overlap=96,
        length_function=len,
    )
    return splitter.split_text(text)


# -----------------------------
# Embeddings (cached)
# -----------------------------
@st.cache_resource
def load_embedder():
    """Cache the embedder so Streamlit hot-reloads don't re-download the model."""
    device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )


# -----------------------------
# Prompts
# -----------------------------
QA_PROMPT = PromptTemplate.from_template(
    "Answer the user's question **only** using the context. "
    "If the answer isn't in the context, say: \"I couldn't find that in the documents.\" "
    "Be concise (2–4 sentences).\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n"
    "Answer:"
)

CONDENSE_PROMPT = PromptTemplate.from_template(
    "Rewrite the follow-up question to be standalone using the chat history.\n"
    "Chat history:\n{chat_history}\n\n"
    "Follow-up question: {question}\n"
    "Standalone question:"
)


# -----------------------------
# Local FLAN-T5 (cached)
# -----------------------------
@st.cache_resource
def load_local_llm():
    """
    Try FLAN-T5-LARGE first (≈770M params).
    If memory issues happen, fall back to FLAN-T5-BASE automatically.
    """
    preferred = "google/flan-t5-large"
    fallback = "google/flan-t5-base"

    # Device + dtype (fp16 on MPS/CUDA; fp32 on CPU)
    if torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float16
    elif torch.cuda.is_available():
        device = "cuda"
        dtype = torch.float16
    else:
        device = "cpu"
        dtype = torch.float32

    def _make_pipeline(model_id: str):
        tok = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_id,
            dtype=(dtype if device != "cpu" else torch.float32),  # <-- use dtype (not torch_dtype)
            low_cpu_mem_usage=True,
        )
        if device != "cpu":
            model.to(device)
        gen = pipeline(
            task="text2text-generation",
            model=model,
            tokenizer=tok,
            max_new_tokens=192,      # leave room for input
            min_new_tokens=48,
            do_sample=False,         # deterministic, better for QA
            num_beams=6,             # quality bump
            length_penalty=1.05,
            no_repeat_ngram_size=3,
        )
        return HuggingFacePipeline(pipeline=gen)

    try:
        llm = _make_pipeline(preferred)
        st.caption(f"LLM ready: local {preferred} (text2text-generation)")
        return llm
    except Exception as e:
        st.warning(f"Couldn't load {preferred} ({e}). Falling back to {fallback}...")
        llm = _make_pipeline(fallback)
        st.caption(f"LLM ready: local {fallback} (text2text-generation)")
        return llm


# -----------------------------
# Vector store + retriever
# -----------------------------
def get_vectorstore(text_chunks):
    embeddings = load_embedder()
    return FAISS.from_texts(texts=text_chunks, embedding=embeddings)


# -----------------------------
# Conversational QA chain
# -----------------------------
def get_conversation_chain(vectorstore):
    llm = load_local_llm()

    # Use MMR to keep 3 diverse, short chunks in context
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 24, "lambda_mult": 0.3},
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history", 
        return_messages=True,
        output_key="answer",
        )

    # IMPORTANT: return sources so we can detect empty retrieval and fallback
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT},
        condense_question_prompt=CONDENSE_PROMPT,
        return_source_documents=True,
        output_key="answer",   # <-- added
    )
    return chain


# -----------------------------
# One-click summarization
# -----------------------------
@st.cache_resource(show_spinner=False)
def _summarizer_llm():
    # reuse the same local model for summarization chain
    return load_local_llm()

def summarize_all_chunks(text_chunks):
    """Map-reduce summary over the whole document using the same local LLM."""
    llm = _summarizer_llm()
    docs = [Document(page_content=t) for t in text_chunks]
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    return chain.run(docs)


# -----------------------------
# UI helpers
# -----------------------------
def _looks_like_summary_request(q: str) -> bool:
    ql = (q or "").lower()
    return any(
        key in ql
        for key in ["what is this about", "what is this pdf about", "summarize", "summary", "overview"]
    )

def handle_userinput(user_question):
    # call with invoke() to avoid deprecation warnings
    resp = st.session_state.conversation.invoke({"question": user_question})
    st.session_state.chat_history = resp.get("chat_history", [])
    answer = resp.get("answer", "")
    sources = resp.get("source_documents") or []

    # If user asked for a summary/what-is-this-about AND retrieval was empty,
    # automatically fall back to a whole-PDF summary.
    retrieval_empty = (len(sources) == 0) or ("I couldn't find that in the documents." in answer)
    if _looks_like_summary_request(user_question) and retrieval_empty:
        if st.session_state.get("text_chunks"):
            with st.spinner("Summarizing the whole PDF..."):
                answer = summarize_all_chunks(st.session_state.text_chunks)

    # render chat history
    for i, message in enumerate(st.session_state.chat_history):
        html = user_template if i % 2 == 0 else bot_template
        st.write(html.replace("{{MSG}}", message.content), unsafe_allow_html=True)

    # also render the (possibly replaced) answer explicitly if history didn’t include it
    if answer and (not st.session_state.chat_history or st.session_state.chat_history[-1].content != answer):
        st.write(bot_template.replace("{{MSG}}", answer), unsafe_allow_html=True)


# -----------------------------
# Streamlit app
# -----------------------------
def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon="📚")
    st.write(css, unsafe_allow_html=True)

    # Reset to avoid stale chains after code/config changes
    with st.sidebar:
        if st.button("Reset app"):
            st.session_state.clear()
            st.experimental_rerun()

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "text_chunks" not in st.session_state:
        st.session_state.text_chunks = None

    st.header("Chat with multiple PDFs 📚")

    # Main question box
    user_question = st.text_input("Ask a question about your documents:")
    if user_question and st.session_state.conversation:
        handle_userinput(user_question)

    # Summarize button (works after Process)
    if st.session_state.get("text_chunks") and st.button("Summarize this PDF"):
        with st.spinner("Summarizing..."):
            summary = summarize_all_chunks(st.session_state.text_chunks)
        st.subheader("Summary")
        st.write(summary)

    # Sidebar upload & processing
    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'",
            accept_multiple_files=True,
            type=["pdf"],
        )
        if st.button("Process", disabled=not pdf_docs):
            with st.spinner("Processing..."):
                # 1) Extract text robustly
                raw_text = get_pdf_text(pdf_docs)

                # 2) Split into chunks
                text_chunks = get_text_chunks(raw_text)

                # 3) Show indexing stats (sanity check)
                total_chars = sum(len(c) for c in text_chunks)
                st.info(f"Indexed {len(text_chunks)} chunks (~{total_chars} characters).")
                if total_chars < 50:
                    st.error(
                        "Very little/no text was indexed. This PDF may be scanned. "
                        "Enable OCR in the code or pre-OCR your PDFs and try again."
                    )
                    st.stop()

                # Stash chunks for summarization
                st.session_state.text_chunks = text_chunks

                # 4) Build vector store
                vectorstore = get_vectorstore(text_chunks)

                # 5) Build conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)

                st.success("Ready! Ask a question above or click 'Summarize this PDF'.")

    if not st.session_state.conversation:
        st.info("Upload PDFs and click **Process** to start.")


if __name__ == "__main__":
    main()
