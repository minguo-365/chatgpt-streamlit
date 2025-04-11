
import streamlit as st
import os
from openai import OpenAI

# 页面配置
st.set_page_config(page_title="ChatGPT GPT-4o 聊天助手", layout="wide")

# 插入公司 Logo
st.image("https://raw.githubusercontent.com/你的用户名/你的仓库名/main/wertgarantie_logo.png", width=160)

# 欢迎话
st.markdown("""
<div style='text-align: center; margin-top: -30px;'>
    <h1>🤖 欢迎使用 ChatGPT GPT-4o 聊天助手</h1>
    <p style='font-size:18px;'>由 <strong>WERTGARANTIE</strong> 提供技术支持，支持中文 ⛄ 和德语 🇩🇪</p>
</div>
""", unsafe_allow_html=True)

# 精美样式
st.markdown("""
<style>
    .stTextInput > div > div > input {
        border-radius: 10px;
        font-size: 18px;
        padding: 10px;
    }
    .stMarkdown {
        font-size: 17px;
        line-height: 1.6;
    }
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# 初始化 OpenAI 客户端
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 聊天记录初始化
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "你是一个友好而智慧的聊天助手"}
    ]

# 显示聊天历史
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
        st.error(f"错误: {e}")
