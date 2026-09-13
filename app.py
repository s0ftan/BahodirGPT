import os
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 1. Настройка страницы
st.set_page_config(page_title="Custom Cloud AI", page_icon="🤖", layout="centered")

# 2. Внедрение стильного CSS-дизайна
st.markdown("""
    <style>
    .stChatMessage {
        border-radius: 16px !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        margin-bottom: 15px !important;
        padding: 15px !important;
        border: 1px solid rgba(0,0,0,0.05);
    }
    .stChatInput textarea {
        border-radius: 24px !important;
        border: 2px solid #e0e0e0 !important;
        padding: 12px 20px !important;
    }
    .stChatInput button {
        border-radius: 50% !important;
        background-color: #10a37f !important;
        color: white !important;
        box-shadow: 0 4px #0d8265 !important;
    }
    .stChatInput button:active {
        box-shadow: 0 1px #0d8265 !important;
        transform: translateY(3px) !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🤖 Мой Премиум Вики-ИИ")
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

# 6. Поле ввода и генерация
if user_question := st.chat_input("Задайте вопрос вашему ИИ..."):
    with st.chat_message("user"):
        st.markdown(user_question)
    st.session_state.messages.append({"role": "user", "content": user_question})

    with st.chat_message("assistant"):
        with st.spinner("ИИ думает..."):
            prompt = f"Context:\n{wiki_context}\n\nQuestion: {user_question}\nAnswer:"
            inputs = tokenizer(prompt, return_tensors="pt")
            
            # Генерация ответа
            outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)
            
            # Декодируем строго первую сгенерированную строку
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Гарантируем, что результат — строка, и очищаем его от системной подсказки
            response_str = str(response)
            clean_response = response_str.replace(prompt, "").strip()
            
            # Финальный вывод на экран
            st.markdown(clean_response)
            
    st.session_state.messages.append({"role": "assistant", "content": clean_response})
