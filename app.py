import os
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 1. Настройка страницы
st.set_page_config(page_title="BahodirGPT", page_icon="🤡", layout="centered")

# 2. Внедрение стильного CSS-дизайна (закругления, 3D-кнопки, тени)
st.markdown("""
    <style>
    /* Стиль для главного контейнера сообщений */
    .stChatMessage {
        border-radius: 16px !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        margin-bottom: 15px !important;
        padding: 15px !important;
        border: 1px solid rgba(0,0,0,0.05);
    }
    
    /* Скругление углов для поля ввода */
    .stChatInput textarea {
        border-radius: 24px !important;
        border: 2px solid #e0e0e0 !important;
        padding: 12px 20px !important;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.03) !important;
        transition: all 0.3s ease;
    }
    .stChatInput textarea:focus {
        border-color: #10a37f !important;
        box-shadow: 0 0 8px rgba(16,163,127,0.2) !important;
    }

    /* Эффект 3D-кнопки отправки */
    .stChatInput button {
        border-radius: 50% !important;
        background-color: #10a37f !important;
        color: white !important;
        box-shadow: 0 4px #0d8265 !important; /* Объемная тень снизу */
        transition: all 0.1s ease !important;
    }
    .stChatInput button:active {
        box-shadow: 0 1px #0d8265 !important; /* Кнопка уходит вниз при нажатии */
        transform: translateY(3px) !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("👻ОТСОСЁТ И ОТЛТЖЕТ🫶")
st.write("Этот ИИ работает на мощных серверах в облаке, поэтому он отвечает мгновенно!")

# 3. ЗАГРУЗКА МОДЕЛИ
@st.cache_resource
def load_ai_model():
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    return tokenizer, model

tokenizer, model = load_ai_model()

# 4. Функция чтения файла знаний
def read_wiki_data():
    if not os.path.exists("wiki_data.txt"):
        return "The capital of France is Paris."
    with open("wiki_data.txt", "r", encoding="utf-8") as f:
        return f.read()

wiki_context = read_wiki_data()

# 5. История сообщений
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 6. Поле ввода
if user_question := st.chat_input("Задайте вопрос вашему ИИ..."):
    with st.chat_message("user"):
        st.markdown(user_question)
    st.session_state.messages.append({"role": "user", "content": user_question})

    with st.chat_message("assistant"):
        with st.spinner("ИИ думает..."):
            prompt = f"Context:\n{wiki_context}\n\nQuestion: {user_question}\nAnswer:"
            inputs = tokenizer(prompt, return_tensors="pt")
            
            outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.3, do_sample=True)
            response = tokenizer.decode(outputs, skip_special_tokens=True)
            clean_response = response.replace(prompt, "").strip()
            
            st.markdown(clean_response)
            
    st.session_state.messages.append({"role": "assistant", "content": clean_response})
