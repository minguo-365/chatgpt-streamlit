
import streamlit as st
import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.set_page_config(page_title="ChatGPT GPT-4o 聊天助手", layout="wide")
st.title("🤖 ChatGPT GPT-4o 聊天助手")

# 聊天记录初始化
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": "你是一个友好而聪明的聊天助手"}]

# 显示对话历史
for msg in st.session_state.messages:
    if msg["role"] != "system":
        st.chat_message(msg["role"]).markdown(msg["content"])

# 用户输入
user_input = st.chat_input("请输入你的问题...")

if user_input:
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=st.session_state.messages
        )
        reply = response.choices[0].message.content
        st.chat_message("assistant").markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})
    except Exception as e:
        st.error(f"出错啦: {str(e)}")
