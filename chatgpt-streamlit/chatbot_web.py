
import openai
import streamlit as st
import os

# 读取 OpenAI API 密钥（从环境变量中）
openai.api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="ChatGPT Web 聊天", layout="wide")
st.title("🤖 ChatGPT GPT-4o 聊天助手")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": "你是一个智能聊天助手"}]

for msg in st.session_state.messages:
    if msg["role"] != "system":
        st.chat_message(msg["role"]).markdown(msg["content"])

user_input = st.chat_input("请输入你的问题...")

if user_input:
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=st.session_state.messages
        )
        reply = response.choices[0].message.content
        st.chat_message("assistant").markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})
    except Exception as e:
        st.error(f"出错啦: {str(e)}")
