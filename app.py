import pandas as pd
import streamlit as st
import plotly.express as px
from langchain.chat_models import ChatOpenAI 
import matplotlib.pyplot as plt
import json
from io import StringIO
import re
from utils.audio_utils import AudioProcessor
import logging
import io
import sounddevice as sd
import soundfile as sf
import numpy as np
import av
import wave
import queue
from sklearn.linear_model import LinearRegression
import threading
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase
import os
import tempfile
import speech_recognition as sr
from gtts import gTTS
from dotenv import load_dotenv
load_dotenv()

# Import custom modules
from utils.audio_utils import AudioProcessor
from utils.data_utils import parse_financial_data, parse_csv_data, analyze_financial_trends, categorize_transactions
from utils.nlp_utils import is_complex_query, handle_unclear_request, extract_date_range, extract_financial_goal
from rag.document_store import DocumentStore
from rag.retriever import RAGRetriever
from agents.financial_agent import FinancialAgent

# Setup logging
logging.basicConfig(level=logging.DEBUG)

# Initialize session state
if 'financial_data' not in st.session_state:
    st.session_state.financial_data = pd.DataFrame()

def safe_rerun():
    """
    Safely rerun the application.
    """
    try:
        st.rerun()
    except Exception as e:
        st.warning(f"Не удалось перезагрузить страницу. Пожалуйста, обновите страницу вручную. Ошибка: {str(e)}")
        st.stop()

class AudioRecorder(AudioProcessorBase):
    """
    Class for capturing audio frames via webrtc_streamer.
    Stores all frames in self.frames while recording=True.
    """
    def __init__(self):
        self.frames = []
        self.recording = False

    def recv(self, frame):
        if self.recording:
            # Convert frame to numpy array and save
            audio_data = frame.to_ndarray()
            self.frames.append(audio_data)
        return frame

class AppState:
    def __init__(self):
        self.financial_data = pd.DataFrame()
        self.chat_history = []
        self.recorder = AudioRecorder()
        self.recording = False
        self.temp_audio = None
        self.show_preview = False
        self.raw_data = None
        self.processed_data = pd.DataFrame()
        self.ai_analysis = {
            "columns": {},
            "metrics": [],
            "categories": [],
            "visualizations": []
        }
        self.audio_processor = AudioProcessor()
        self.document_store = DocumentStore()
        # Используем DeepSeek API через OpenAI совместимый интерфейс
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")
        
        # Настраиваем модель с помощью LangChain
        self.llm = ChatOpenAI(
            openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
            openai_api_base="https://api.deepseek.com",
            model_name="deepseek-chat",
            temperature=0.7
        )
        self.rag_retriever = RAGRetriever(self.llm, self.document_store)
        self.financial_agent = FinancialAgent(self.llm)

# Initialize application state
if 'app_state' not in st.session_state:
    st.session_state.app_state = AppState()

@st.cache_data
def analyze_data_with_ai(data_input):
    """
    Analyze data using AI.
    
    Args:
        data_input: Data to analyze
        
    Returns:
        Analysis results
    """
    analysis_prompt = f"""
    Проанализируй следующие CSV данные, которые могут содержать большое количество информации по разным категориям.
    Твоя задача:
    1. Отопредели типы данных каждого столбца (например, datetime, numeric, category, text).
    2. Отфильтруй ненужные данные и оставь только те столбцы, которые пригодны для статистической обработки и построения диаграмм.
    3. Сформируй рекомендации по визуализациям, адаптированные к структуре данных.

    Верни ТОЛЬКО JSON с такой структурой:
    {{
        "columns": {{"название_столбца": "тип", ...}},
        "metrics": ["имена числовых столбцов"],
        "categories": ["имена категориальных столбцов"],
        "visualizations": [
            {{
                "type": "тип диаграммы",
                "x": "название столбца для оси X",
                "y": "название столбца для оси Y",
                "desc": "описание"
            }}
        ]
    }}

    Данные (первые 2000 символов):
    {data_input[:2000]}
    """
    
    raw_response = st.session_state.app_state.rag_retriever.retrieve_and_generate(analysis_prompt)["response"]
    
    try:
        json_str = re.search(r'\{.*\}', raw_response, re.DOTALL).group()
        return json.loads(json_str)
    except (AttributeError, json.JSONDecodeError) as e:
        st.error(f"Ошибка при анализе данных: {str(e)}")
        return {
            "columns": {},
            "metrics": [],
            "categories": [],
            "visualizations": []
        }

# Page configuration
st.set_page_config(page_title="Финансовый помощник", layout="wide", page_icon="💰")
st.markdown(
    """
    <style>
    .big-font {
        font-size: 40px !important;
        font-weight: bold;
        text-align: center;
        color: #2E86C1;
    }
    .medium-font {
        font-size: 20px !important;
        text-align: justify;
        color: #34495E;
    }
    .stButton button {
        background-color: #2E86C1;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 10px 20px;
    }
    .stButton button:hover {
        background-color: #3498DB;
    }
    .stFileUploader div {
        border: 2px dashed #2E86C1;
        border-radius: 10px;
        padding: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

def main_page():
    """Display the application's main page"""
    st.markdown('<p class="big-font">Помощник для управления личными финансами и планирования бюджета</p>',
                unsafe_allow_html=True)
    st.markdown(
        """
        <p class="medium-font">
        Добро пожаловать в ваш персональный финансовый помощник! 🚀<br><br>
        Это приложение поможет вам взять под контроль свои финансы, научиться грамотно распределять бюджет 
        и достигать финансовых целей. Мы предлагаем простые и эффективные инструменты для управления 
        вашими доходами и расходами, анализа финансовых привычек и планирования будущего.
        </p>
        """,
        unsafe_allow_html=True,
    )
    
    st.subheader("✨ Ключевые функции")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            """
            <div class="feature-card">
                <h3>📊 Отслеживание расходов</h3>
                <p>Регулярно анализируйте свои траты и находите возможности для экономии.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div class="feature-card">
                <h3>📅 Планирование бюджета</h3>
                <p>Создавайте реалистичные бюджеты и следите за их выполнением.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            """
            <div class="feature-card">
                <h3>🎯 Финансовые цели</h3>
                <p>Ставьте цели и отслеживайте прогресс в их достижении.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
    st.subheader("📋 Примеры использования")
    st.write("Вот как вы можете использовать наше приложение:")
    example_data = pd.DataFrame({
        "Категория": ["Еда", "Транспорт", "Развлечения", "Жилье"],
        "Сумма": [15000, 5000, 7000, 20000]
    })
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Расходы по категориям**")
        st.bar_chart(example_data.set_index("Категория"))
    with col2:
        st.write("**Распределение бюджета**")
        fig, ax = plt.subplots()
        example_data.groupby("Категория")["Сумма"].sum().plot(kind="pie", autopct="%1.1f%%", ax=ax)
        st.pyplot(fig)
        
    st.subheader("💬 Отзывы пользователей")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            """
            <div class="testimonial-card">
                <h4>Анна, 28 лет</h4>
                <p>"Это приложение изменило мой подход к финансам. Теперь я точно знаю, куда уходят мои деньги!"</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div class="testimonial-card">
                <h4>Иван, 35 лет</h4>
                <p>"Благодаря этому приложению я смог накопить на отпуск всего за 4 месяца!"</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            """
            <div class="testimonial-card">
                <h4>Елена, 42 года</h4>
                <p>"Отличный помощник в планировании семейного бюджета. Рекомендую всем!"</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

def data_input_page():
    """Display the data input page"""
    st.markdown('<p class="big-font">Ввод финансовых данных</p>', unsafe_allow_html=True)
    st.write("Загрузите данные о своих финансах или введите их вручную.")
    
    tab1, tab2 = st.tabs(["Загрузка файла", "Ручной ввод"])
    
    with tab1:
        st.write("Загрузите CSV-файл с вашими финансовыми данными.")
        uploaded_file = st.file_uploader("Выберите CSV-файл", type="csv")
        
        if uploaded_file is not None:
            df = parse_csv_data(uploaded_file)
            if df is not None:
                st.session_state.app_state.financial_data = df
                st.session_state.financial_data = df
                st.success("Данные успешно загружены!")
                st.dataframe(df.head())
                
                # Add data to document store for RAG retrieval
                st.session_state.app_state.document_store.add_financial_data(df)
                
                # Redirect to analyze page
                st.session_state.page = "analyze"
                safe_rerun()
    
    with tab2:
        st.write("Введите ваши финансовые данные в следующем формате:")
        st.code("""
        Доходы:
        - Зарплата: 80000
        - Фриланс: 20000

        Расходы:
        - Аренда: 30000
        - Продукты: 15000
        - Транспорт: 5000
        - Развлечения: 10000
        """)
        
        text_data = st.text_area("Введите ваши данные:", height=300)
        
        if st.button("Анализировать данные"):
            if text_data:
                df = parse_financial_data(text_data)
                if not df.empty:
                    st.session_state.app_state.financial_data = df
                    st.session_state.financial_data = df
                    st.success("Данные успешно обработаны!")
                    st.dataframe(df)
                    
                    # Add data to document store for RAG retrieval
                    st.session_state.app_state.document_store.add_financial_data(df)
                    
                    # Redirect to analyze page
                    st.session_state.page = "analyze"
                    safe_rerun()
                else:
                    st.error("Не удалось распознать данные. Проверьте формат ввода.")
            else:
                st.warning("Пожалуйста, введите данные.")

def analyze_page():
    """Display the data analysis page"""
    st.markdown('<p class="big-font">Анализ финансов</p>', unsafe_allow_html=True)
    
    if st.session_state.app_state.financial_data.empty:
        st.warning("Нет данных для анализа. Пожалуйста, загрузите или введите ваши финансовые данные.")
        if st.button("Ввести данные"):
            st.session_state.page = "data_input"
            safe_rerun()
        return
    
    # Show basic statistics
    st.subheader("Обзор данных")
    st.dataframe(st.session_state.app_state.financial_data)
    
    # Create visualizations
    st.subheader("Визуализация данных")
    
    # Determine if we have income and expense data
    if "Тип" in st.session_state.app_state.financial_data.columns:
        # Create income and expense summary
        income_data = st.session_state.app_state.financial_data[st.session_state.app_state.financial_data["Тип"] == "Доходы"]
        expense_data = st.session_state.app_state.financial_data[st.session_state.app_state.financial_data["Тип"] == "Расходы"]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Структура доходов**")
            if not income_data.empty and "Категория" in income_data.columns:
                fig = px.pie(income_data, values="Сумма", names="Категория", title="Доходы по категориям")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Нет данных о доходах по категориям.")
                
        with col2:
            st.write("**Структура расходов**")
            if not expense_data.empty and "Категория" in expense_data.columns:
                fig = px.pie(expense_data, values="Сумма", names="Категория", title="Расходы по категориям")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Нет данных о расходах по категориям.")
        
        # Create budget summary
        if not income_data.empty and not expense_data.empty:
            st.subheader("Бюджетная сводка")
            total_income = income_data["Сумма"].sum()
            total_expenses = expense_data["Сумма"].sum()
            balance = total_income - total_expenses
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Общий доход", f"{total_income:,.0f} ₽")
            col2.metric("Общие расходы", f"{total_expenses:,.0f} ₽")
            col3.metric("Баланс", f"{balance:,.0f} ₽", f"{balance/total_income*100:.1f}%" if total_income > 0 else "0%")
            
            # Budget bar chart
            budget_data = pd.DataFrame({
                "Категория": ["Доходы", "Расходы", "Баланс"],
                "Сумма": [total_income, total_expenses, balance]
            })
            
            fig = px.bar(budget_data, x="Категория", y="Сумма", title="Обзор бюджета",
                         color="Категория", color_discrete_map={"Доходы": "green", "Расходы": "red", "Баланс": "blue"})
            st.plotly_chart(fig, use_container_width=True)
    else:
        # Simple visualization for data without income/expense classification
        if "Категория" in st.session_state.app_state.financial_data.columns and "Сумма" in st.session_state.app_state.financial_data.columns:
            st.write("**Распределение по категориям**")
            fig = px.pie(st.session_state.app_state.financial_data, values="Сумма", names="Категория", title="Распределение по категориям")
            st.plotly_chart(fig, use_container_width=True)
            
            st.write("**Суммы по категориям**")
            fig = px.bar(st.session_state.app_state.financial_data.groupby("Категория")["Сумма"].sum().reset_index(),
                         x="Категория", y="Сумма", title="Суммы по категориям")
            st.plotly_chart(fig, use_container_width=True)
    
    # AI Analysis
    st.subheader("ИИ-анализ данных")
    if st.button("Провести ИИ-анализ"):
        with st.spinner("Анализируем ваши данные с помощью ИИ..."):
            csv_buffer = StringIO()
            st.session_state.app_state.financial_data.to_csv(csv_buffer)
            csv_data = csv_buffer.getvalue()
            
            st.session_state.app_state.ai_analysis = analyze_data_with_ai(csv_data)
            
            st.success("Анализ завершен!")
            
            # Display AI findings
            if st.session_state.app_state.ai_analysis["visualizations"]:
                st.write("**Рекомендуемые визуализации:**")
                for viz in st.session_state.app_state.ai_analysis["visualizations"]:
                    st.write(f"- {viz['desc']}")
                    
                    # Create the recommended visualization if columns exist
                    if viz['x'] in st.session_state.app_state.financial_data.columns and viz['y'] in st.session_state.app_state.financial_data.columns:
                        if viz['type'].lower() == 'bar':
                            fig = px.bar(st.session_state.app_state.financial_data, x=viz['x'], y=viz['y'], title=viz['desc'])
                            st.plotly_chart(fig, use_container_width=True)
                        elif viz['type'].lower() == 'pie':
                            fig = px.pie(st.session_state.app_state.financial_data, values=viz['y'], names=viz['x'], title=viz['desc'])
                            st.plotly_chart(fig, use_container_width=True)
                        elif viz['type'].lower() == 'line':
                            fig = px.line(st.session_state.app_state.financial_data, x=viz['x'], y=viz['y'], title=viz['desc'])
                            st.plotly_chart(fig, use_container_width=True)
                        elif viz['type'].lower() == 'scatter':
                            fig = px.scatter(st.session_state.app_state.financial_data, x=viz['x'], y=viz['y'], title=viz['desc'])
                            st.plotly_chart(fig, use_container_width=True)
    
    # Financial trends analysis
    if "Дата" in st.session_state.app_state.financial_data.columns and "Сумма" in st.session_state.app_state.financial_data.columns:
        st.subheader("Анализ трендов")
        trend_data = analyze_financial_trends(st.session_state.app_state.financial_data)
        
        if trend_data["trends"]:
            st.write("**Обнаруженные тренды:**")
            for trend in trend_data["trends"]:
                st.write(f"- Категория '{trend['category']}': {trend['direction']} (уверенность: {trend['confidence']:.2f})")
        
        if trend_data["projections"]:
            st.write("**Прогнозы на следующий период:**")
            projection_data = pd.DataFrame({
                "Категория": list(trend_data["projections"].keys()),
                "Прогноз": list(trend_data["projections"].values())
            })
            st.dataframe(projection_data)
            
            fig = px.bar(projection_data, x="Категория", y="Прогноз", title="Прогноз на следующий период")
            st.plotly_chart(fig, use_container_width=True)

def assistant_page():
    """Display the financial assistant interaction page"""
    st.markdown('<p class="big-font">Финансовый ассистент</p>', unsafe_allow_html=True)
    
    # Display tabs for text and voice interfaces
    tab1, tab2 = st.tabs(["Текстовый чат", "Голосовой ассистент"])
    
    with tab1:
        st.write("Задайте вопрос о ваших финансах, получите совет или попросите выполнить анализ.")
        
        # Check if we have financial data
        if st.session_state.app_state.financial_data.empty:
            st.warning("У вас еще нет загруженных финансовых данных. Некоторые функции могут быть недоступны.")
            if st.button("Загрузить данные"):
                st.session_state.page = "data_input"
                safe_rerun()
        
        # User input
        user_query = st.text_input("Ваш запрос:")
        
        # Process user query
        if user_query:
            with st.spinner("Обрабатываю ваш запрос..."):
                # Check for unclear requests
                unclear_response = handle_unclear_request(user_query)
                if unclear_response:
                    st.session_state.app_state.chat_history.append({"role": "user", "content": user_query})
                    st.session_state.app_state.chat_history.append({"role": "assistant", "content": unclear_response})
                # Check if it's a complex query
                elif is_complex_query(user_query):
                    # Use financial agent for complex queries
                    response = st.session_state.app_state.financial_agent.handle_complex_query(
                        user_query, st.session_state.app_state.financial_data
                    )
                    
                    # Format response for chat
                    formatted_response = f"**{response['task_type']}**\n\n{response['analysis']}\n\n**Рекомендации:**\n"
                    for rec in response['recommendations']:
                        formatted_response += f"- {rec}\n"
                    
                    st.session_state.app_state.chat_history.append({"role": "user", "content": user_query})
                    st.session_state.app_state.chat_history.append({"role": "assistant", "content": formatted_response})
                    
                    # Display visualizations if available
                    if response["visualizations"]:
                        for viz in response["visualizations"]:
                            # Check if we have the required columns
                            x_col = viz["data"]["x"]
                            y_col = viz["data"]["y"]
                            
                            if x_col in st.session_state.app_state.financial_data.columns and y_col in st.session_state.app_state.financial_data.columns:
                                if viz["type"].lower() == "pie":
                                    fig = px.pie(
                                        st.session_state.app_state.financial_data, 
                                        values=y_col, 
                                        names=x_col, 
                                        title=viz["title"]
                                    )
                                    st.plotly_chart(fig)
                                elif viz["type"].lower() in ["bar", "column"]:
                                    fig = px.bar(
                                        st.session_state.app_state.financial_data, 
                                        x=x_col, 
                                        y=y_col, 
                                        title=viz["title"]
                                    )
                                    st.plotly_chart(fig)
                                elif viz["type"].lower() == "line":
                                    fig = px.line(
                                        st.session_state.app_state.financial_data, 
                                        x=x_col, 
                                        y=y_col, 
                                        title=viz["title"]
                                    )
                                    st.plotly_chart(fig)
                else:
                    # Use RAG for normal queries
                    rag_response = st.session_state.app_state.rag_retriever.retrieve_and_generate(
                        user_query, st.session_state.app_state.financial_data
                    )
                    
                    st.session_state.app_state.chat_history.append({"role": "user", "content": user_query})
                    st.session_state.app_state.chat_history.append({"role": "assistant", "content": rag_response["response"]})
        
        # Display chat history
        st.subheader("История диалога")
        for message in st.session_state.app_state.chat_history:
            if message["role"] == "user":
                st.markdown(f"**Вы:** {message['content']}")
            else:
                st.markdown(f"**Ассистент:** {message['content']}")
        
        # Clear chat button
        if st.button("Очистить историю"):
            st.session_state.app_state.chat_history = []
            safe_rerun()
    
    with tab2:
        st.write("Общайтесь с ассистентом с помощью голоса.")
        
        # Initialize webrtc streamer for audio recording
        webrtc_ctx = webrtc_streamer(
            key="voice-assistant",
            mode=WebRtcMode.SENDONLY,
            audio_processor_factory=lambda: st.session_state.app_state.recorder,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        )
        
        # Recording controls
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🎤 Начать запись" if not st.session_state.app_state.recording else "⏹️ Остановить запись"):
                st.session_state.app_state.recording = not st.session_state.app_state.recording
                st.session_state.app_state.recorder.recording = st.session_state.app_state.recording
                
                if not st.session_state.app_state.recording and st.session_state.app_state.recorder.frames:
                    # Convert frames to WAV bytes when stopping recording
                    audio_bytes = st.session_state.app_state.audio_processor.frames_to_wav_bytes(
                        st.session_state.app_state.recorder.frames
                    )
                    st.session_state.app_state.temp_audio = audio_bytes
                    st.session_state.app_state.show_preview = True
                    
                    # Process audio with DeepSpeech
                    with st.spinner("Распознаю речь..."):
                        transcribed_text = st.session_state.app_state.audio_processor.speech_to_text(audio_bytes)
                        
                        if transcribed_text and transcribed_text != "Ошибка распознавания речи":
                            st.success(f"Распознано: {transcribed_text}")
                            
                            # Process the transcribed query
                            with st.spinner("Обрабатываю ваш запрос..."):
                                # Check for unclear requests
                                unclear_response = handle_unclear_request(transcribed_text)
                                if unclear_response:
                                    response_text = unclear_response
                                # Check if it's a complex query
                                elif is_complex_query(transcribed_text):
                                    # Use financial agent for complex queries
                                    agent_response = st.session_state.app_state.financial_agent.handle_complex_query(
                                        transcribed_text, st.session_state.app_state.financial_data
                                    )
                                    
                                    # Format response for voice
                                    response_text = f"{agent_response['analysis']} Мои рекомендации: "
                                    for rec in agent_response['recommendations'][:2]:  # Limit to 2 recommendations for voice
                                        response_text += f"{rec}. "
                                else:
                                    # Use RAG for normal queries
                                    rag_response = st.session_state.app_state.rag_retriever.retrieve_and_generate(
                                        transcribed_text, st.session_state.app_state.financial_data
                                    )
                                    response_text = rag_response["response"]
                                
                                # Convert response to speech
                                with st.spinner("Генерирую ответ..."):
                                    response_audio, sample_rate = st.session_state.app_state.audio_processor.text_to_speech(response_text)
                                    
                                    # Add to chat history
                                    st.session_state.app_state.chat_history.append({"role": "user", "content": transcribed_text})
                                    st.session_state.app_state.chat_history.append({"role": "assistant", "content": response_text})
                                    
                                    # Play audio response
                                    st.audio(response_audio, format="audio/wav")
                                    
                                    # Clear recording
                                    st.session_state.app_state.recorder.frames = []
                                    safe_rerun()
                        else:
                            st.error("Не удалось распознать речь. Пожалуйста, попробуйте снова.")
                            st.session_state.app_state.recorder.frames = []
        
        with col2:
            if st.session_state.app_state.recording:
                st.info("🔴 Идет запись... Говорите в микрофон.")
        
        # Show audio recording preview if available
        if st.session_state.app_state.show_preview and st.session_state.app_state.temp_audio:
            st.audio(st.session_state.app_state.temp_audio, format="audio/wav")
            
            if st.button("Очистить аудио"):
                st.session_state.app_state.temp_audio = None
                st.session_state.app_state.show_preview = False
                st.session_state.app_state.recorder.frames = []
                safe_rerun()

def budget_planning_page():
    """Display the budget planning page"""
    st.markdown('<p class="big-font">Планирование бюджета</p>', unsafe_allow_html=True)
    
    if st.session_state.app_state.financial_data.empty:
        st.warning("Нет данных для планирования бюджета. Пожалуйста, загрузите или введите ваши финансовые данные.")
        if st.button("Ввести данные"):
            st.session_state.page = "data_input"
            safe_rerun()
        return
    
    # Create budget planning tabs
    tab1, tab2 = st.tabs(["Месячный бюджет", "Долгосрочное планирование"])
    
    with tab1:
        st.write("Создайте план бюджета на месяц на основе ваших финансовых данных.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Income data input
            st.subheader("Доходы")
            
            # Extract existing income data if available
            income_categories = []
            income_amounts = []
            
            if "Тип" in st.session_state.app_state.financial_data.columns:
                income_data = st.session_state.app_state.financial_data[st.session_state.app_state.financial_data["Тип"] == "Доходы"]
                if not income_data.empty and "Категория" in income_data.columns:
                    for category, group in income_data.groupby("Категория"):
                        income_categories.append(category)
                        income_amounts.append(group["Сумма"].sum())
            
            # Allow modifying income data
            for i, category in enumerate(income_categories):
                st.text_input(f"Категория дохода {i+1}", value=category, key=f"income_cat_{i}")
                st.number_input(f"Сумма {i+1}", value=float(income_amounts[i]), key=f"income_amount_{i}")
            
            # Add new income category
            st.text_input("Новая категория дохода", key="new_income_cat")
            st.number_input("Сумма", value=0.0, key="new_income_amount")
            
            if st.button("Добавить категорию дохода"):
                if st.session_state["new_income_cat"] and st.session_state["new_income_amount"] > 0:
                    new_row = pd.DataFrame({
                        "Категория": [st.session_state["new_income_cat"]],
                        "Тип": ["Доходы"],
                        "Сумма": [st.session_state["new_income_amount"]]
                    })
                    st.session_state.app_state.financial_data = pd.concat([st.session_state.app_state.financial_data, new_row])
                    st.success(f"Добавлена категория дохода: {st.session_state['new_income_cat']}")
                    safe_rerun()
                else:
                    st.error("Введите название категории и сумму больше 0.")
        
        with col2:
            # Expense data input
            st.subheader("Расходы")
            
            # Extract existing expense data if available
            expense_categories = []
            expense_amounts = []
            
            if "Тип" in st.session_state.app_state.financial_data.columns:
                expense_data = st.session_state.app_state.financial_data[st.session_state.app_state.financial_data["Тип"] == "Расходы"]
                if not expense_data.empty and "Категория" in expense_data.columns:
                    for category, group in expense_data.groupby("Категория"):
                        expense_categories.append(category)
                        expense_amounts.append(group["Сумма"].sum())
            
            # Allow modifying expense data
            for i, category in enumerate(expense_categories):
                st.text_input(f"Категория расхода {i+1}", value=category, key=f"expense_cat_{i}")
                st.number_input(f"Сумма {i+1}", value=float(expense_amounts[i]), key=f"expense_amount_{i}")
            
            # Add new expense category
            st.text_input("Новая категория расхода", key="new_expense_cat")
            st.number_input("Сумма", value=0.0, key="new_expense_amount")
            
            if st.button("Добавить категорию расхода"):
                if st.session_state["new_expense_cat"] and st.session_state["new_expense_amount"] > 0:
                    new_row = pd.DataFrame({
                        "Категория": [st.session_state["new_expense_cat"]],
                        "Тип": ["Расходы"],
                        "Сумма": [st.session_state["new_expense_amount"]]
                    })
                    st.session_state.app_state.financial_data = pd.concat([st.session_state.app_state.financial_data, new_row])
                    st.success(f"Добавлена категория расхода: {st.session_state['new_expense_cat']}")
                    safe_rerun()
                else:
                    st.error("Введите название категории и сумму больше 0.")
        
        # Generate budget plan
        if st.button("Сгенерировать план бюджета"):
            with st.spinner("Создаю план бюджета..."):
                budget_plan = st.session_state.app_state.financial_agent.create_budget_plan(st.session_state.app_state.financial_data)
                
                st.subheader("План бюджета")
                
                # Display budget summary
                st.write(f"**Краткое описание:** {budget_plan['summary']}")
                
                # Display income and expenses
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Доходы:**")
                    income_df = pd.DataFrame({
                        "Категория": budget_plan['income']['categories'],
                        "Сумма": budget_plan['income']['amounts']
                    })
                    st.dataframe(income_df)
                    
                    # Income pie chart
                    fig = px.pie(income_df, values="Сумма", names="Категория", title="Структура доходов")
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.write("**Расходы:**")
                    expense_df = pd.DataFrame({
                        "Категория": budget_plan['expenses']['categories'],
                        "Сумма": budget_plan['expenses']['amounts'],
                        "Необходимые": budget_plan['expenses']['is_necessary']
                    })
                    st.dataframe(expense_df)
                    
                    # Expense pie chart
                    fig = px.pie(expense_df, values="Сумма", names="Категория", title="Структура расходов")
                    st.plotly_chart(fig, use_container_width=True)
                
                # Display savings
                st.metric("Сбережения", f"{budget_plan['savings']:,.0f} ₽")
                
                # Display recommendations
                st.subheader("Рекомендации")
                for rec in budget_plan['recommendations']:
                    st.write(f"- {rec}")
    
    with tab2:
        st.write("Планирование долгосрочных финансовых целей.")
        
        # Goal input
        st.subheader("Финансовая цель")
        goal_name = st.text_input("Название цели", "Отпуск")
        target_amount = st.number_input("Целевая сумма", min_value=0.0, value=100000.0)
        timeframe = st.slider("Срок достижения (месяцы)", min_value=1, max_value=60, value=12)
        
        # Calculate required monthly savings
        monthly_savings = target_amount / timeframe
        
        # Current financial state
        if "Тип" in st.session_state.app_state.financial_data.columns:
            income_data = st.session_state.app_state.financial_data[st.session_state.app_state.financial_data["Тип"] == "Доходы"]
            expense_data = st.session_state.app_state.financial_data[st.session_state.app_state.financial_data["Тип"] == "Расходы"]
            
            total_income = income_data["Сумма"].sum() if not income_data.empty else 0
            total_expenses = expense_data["Сумма"].sum() if not expense_data.empty else 0
            current_savings = total_income - total_expenses
        else:
            current_savings = 0
        
        # Calculate feasibility
        feasibility = min(100, max(0, (current_savings / monthly_savings) * 100))
        
        # Display goal info
        st.subheader("Анализ цели")
        col1, col2, col3 = st.columns(3)
        col1.metric("Требуемые ежемесячные сбережения", f"{monthly_savings:,.0f} ₽")
        col2.metric("Текущие возможные сбережения", f"{current_savings:,.0f} ₽")
        col3.metric("Выполнимость", f"{feasibility:.1f}%")
        
        # Progress visualization
        st.subheader("Прогресс достижения цели")
        
        # Generate time periods
        periods = list(range(1, timeframe + 1))
        accumulated = [monthly_savings * i for i in periods]
        
        # Create DataFrame for chart
        progress_df = pd.DataFrame({
            "Месяц": periods,
            "Накопления": accumulated,
            "Цель": [target_amount] * timeframe
        })
        
        fig = px.line(progress_df, x="Месяц", y=["Накопления", "Цель"], 
                     title=f"Прогресс накоплений для цели '{goal_name}'")
        st.plotly_chart(fig, use_container_width=True)
        
        # Recommendations
        st.subheader("Рекомендации")
        
        if current_savings < monthly_savings:
            shortfall = monthly_savings - current_savings
            st.warning(f"Для достижения цели вам нужно увеличить ежемесячные сбережения на {shortfall:,.0f} ₽")
            
            # Get recommendations from financial agent
            with st.spinner("Получаю рекомендации..."):
                recommendations_prompt = f"""
                Пользователь хочет достичь финансовой цели '{goal_name}' с суммой {target_amount} ₽ за {timeframe} месяцев.
                Текущий доход: {total_income} ₽.
                Текущие расходы: {total_expenses} ₽.
                Текущие возможные сбережения: {current_savings} ₽.
                Требуемые ежемесячные сбережения: {monthly_savings} ₽.
                Не хватает: {shortfall} ₽ в месяц.
                
                Дай 3-5 конкретных рекомендаций, как пользователь может увеличить сбережения или сократить расходы,
                чтобы достичь своей цели в указанный срок.
                """
                
                response = st.session_state.app_state.rag_retriever.retrieve_and_generate(recommendations_prompt)["response"]
                st.write(response)
        else:
            st.success(f"Вы можете достичь своей цели за {timeframe} месяцев при текущем уровне сбережений!")
            
            # Suggest faster goal achievement
            potential_timeframe = int(target_amount / current_savings)
            if potential_timeframe < timeframe:
                st.info(f"При вашем текущем уровне сбережений вы можете достичь цели всего за {potential_timeframe} месяцев вместо {timeframe}!")

def main():
    # Initialize session state
    if 'page' not in st.session_state:
        st.session_state.page = "main"
    
    # Sidebar navigation
    st.sidebar.title("Навигация")
    
    if st.sidebar.button("Главная"):
        st.session_state.page = "main"
        safe_rerun()
        
    if st.sidebar.button("Ввод данных"):
        st.session_state.page = "data_input"
        safe_rerun()
        
    if st.sidebar.button("Анализ"):
        st.session_state.page = "analyze"
        safe_rerun()
        
    if st.sidebar.button("Ассистент"):
        st.session_state.page = "assistant"
        safe_rerun()
        
    if st.sidebar.button("Планирование бюджета"):
        st.session_state.page = "budget_planning"
        safe_rerun()
    
    # Display current page
    if st.session_state.page == "main":
        main_page()
    elif st.session_state.page == "data_input":
        data_input_page()
    elif st.session_state.page == "analyze":
        analyze_page()
    elif st.session_state.page == "assistant":
        assistant_page()
    elif st.session_state.page == "budget_planning":
        budget_planning_page()

if __name__ == "__main__":
    main()
