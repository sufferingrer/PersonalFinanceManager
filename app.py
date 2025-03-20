import pandas as pd
import streamlit as st
import plotly.express as px
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
import matplotlib.pyplot as plt
import json
from io import StringIO
import re
import logging
from gtts import gTTS
import io
import sounddevice as sd
import soundfile as sf
import numpy as np
import av
import speech_recognition as sr
import wave
import queue
import threading
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase

if 'financial_data' not in st.session_state:
    st.session_state.financial_data = pd.DataFrame()

logging.basicConfig(level=logging.DEBUG)


# Парсеры данных
def parse_financial_data(text):
    """Парсинг произвольного текста в DataFrame"""
    data = []
    current_type = ""
    for line in text.split('\n'):
        line = line.strip()
        type_match = re.match(r'^([А-Яа-я]+)[:\s]', line)
        if type_match:
            current_type = type_match.group(1)
            continue
        if current_type and line.startswith('-'):
            match = re.split(r':\s*', line[1:].strip(), maxsplit=1)
            if len(match) == 2:
                category = match[0].strip()
                amount = re.sub(r'[^\d]', '', match[1])
                if amount.isdigit():
                    data.append({
                        'Категория': category,
                        'Тип': current_type,
                        'Сумма': int(amount)
                    })
    return pd.DataFrame(data)


def safe_rerun():
    """
    Безопасная функция для перезапуска приложения.
    Если в версии Streamlit нет experimental_rerun, просто останавливаем скрипт.
    """
    if hasattr(st, "experimental_rerun"):
        st.experimental_rerun()
    else:
        st.warning("Перезагрузите страницу вручную: в вашей версии Streamlit нет experimental_rerun.")
        st.stop()


class AudioRecorder(AudioProcessorBase):
    """
    Класс для захвата аудиокадров через webrtc_streamer.
    Сохраняет все кадры в self.frames, пока recording=True.
    """

    def __init__(self):
        self.frames = []
        self.recording = False

    def recv(self, frame):
        if self.recording:
            # Преобразуем фрейм в numpy-массив и сохраняем
            audio_data = frame.to_ndarray()
            self.frames.append(audio_data)
        return frame

    def to_wav_bytes(self, sample_rate=16000):
        """
        Превращает накопленные аудиокадры в WAV-байты и возвращает их.
        После этого self.frames можно очистить или оставить для повторного использования.
        """
        with io.BytesIO() as bio:
            with wave.open(bio, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 16 бит
                wf.setframerate(sample_rate)
                for frame_data in self.frames:
                    wf.writeframes(frame_data.tobytes())
            return bio.getvalue()


def audio_to_text(audio_bytes):
    """
    Распознавание речи с помощью Google API.
    Возвращает текст или сообщение об ошибке.
    """
    r = sr.Recognizer()
    with io.BytesIO(audio_bytes) as bio:
        with sr.AudioFile(bio) as source:
            audio = r.record(source)
            try:
                return r.recognize_google(audio, language="ru-RU")
            except sr.UnknownValueError:
                return "Не удалось распознать речь"
            except sr.RequestError:
                return "Ошибка сервиса распознавания"


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


# Инициализация состояния приложения
if 'app_state' not in st.session_state:
    st.session_state.app_state = AppState()


@st.cache_data
def analyze_data_with_ai(data_input):
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
    raw_response = LLMChain(
        llm=llm,
        prompt=PromptTemplate.from_template("{prompt}")
    ).run(prompt=analysis_prompt)
    json_str = re.search(r'\{.*\}', raw_response, re.DOTALL).group()
    return json.loads(json_str)


def parse_csv_data(uploaded_file):
    """Универсальное чтение CSV-файлов"""
    try:
        df = pd.read_csv(
            uploaded_file,
            sep=None,
            engine='python',
            encoding='utf-8',
            thousands=',',
            dtype={'Сумма': 'Int64'},
            on_bad_lines='warn'
        )
        return df
    except UnicodeDecodeError:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, encoding='latin1')
        return df
    except pd.errors.EmptyDataError:
        st.error("Файл пуст. Пожалуйста, загрузите файл с данными.")
        return None
    except pd.errors.ParserError:
        st.error("Ошибка при чтении файла. Проверьте формат файла.")
        return None
    except Exception as e:
        st.error(f"Ошибка чтения файла: {str(e)}")
        return None


# Инициализация LLM
GROQ_API_KEY = "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"
llm = ChatGroq(api_key=GROQ_API_KEY, model_name="llama3-70b-8192")


def is_complex_query(query: str) -> bool:
    complex_keywords = ["бюджет", "анализ", "риск", "инвестиции"]
    return any(keyword in query.lower() for keyword in complex_keywords)


def handle_unclear_request(query):
    unclear_phrases = ["не знаю", "пример", "помощь"]
    if any(phrase in query.lower() for phrase in unclear_phrases):
        return "Пожалуйста, уточните ваш запрос..."
    return None


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
    st.markdown(
        """
        <style>
        .big-font {
            font-size: 48px !important;
            font-weight: bold;
            text-align: center;
            color: #2E86C1;
            margin-bottom: 20px;
        }
        .medium-font {
            font-size: 20px !important;
            text-align: center;
            color: #34495E;
            margin-bottom: 40px;
        }
        .feature-card {
            background: linear-gradient(135deg, #f5f7fa, #c3cfe2);
            border-radius: 15px;
            padding: 20px;
            margin: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            transition: transform 0.3s, box-shadow 0.3s;
        }
        .feature-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
        }
        .testimonial-card {
            background: #ffffff;
            border-radius: 15px;
            padding: 20px;
            margin: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        .cta-button {
            background: linear-gradient(135deg, #2E86C1, #3498DB);
            color: white;
            font-weight: bold;
            border-radius: 25px;
            padding: 15px 30px;
            font-size: 18px;
            border: none;
            cursor: pointer;
            transition: transform 0.3s, box-shadow 0.3s;
        }
        .cta-button:hover {
            transform: scale(1.05);
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
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
                <p>"Очень удобно планировать бюджет и видеть прогресс в достижении целей."</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            """
            <div class="testimonial-card">
                <h4>Мария, 42 года</h4>
                <p>"Советы по экономии помогли мне сократить расходы на 20%!"</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(
            """
            <button class="cta-button">Начать сейчас</button>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Начать сейчас", key="start_button_unique"):
            st.success("Добро пожаловать! Переходите к разделу 'Мой профиль', чтобы начать.")


def profile_page():
    st.header("📁 Мой профиль")
    with st.form("profile_form"):
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Имя")
            surname = st.text_input("Фамилия")
            age = st.number_input("Возраст", min_value=0, max_value=120)
        with col2:
            gender = st.selectbox("Пол", ["Мужской", "Женский", "Другой"])
            phone = st.text_input("Номер телефона")
            email = st.text_input("Электронная почта")
        submitted = st.form_submit_button("Сохранить")
        if submitted:
            st.success("Данные успешно сохранены!")


def reports_page():
    st.header("📊 Анализ отчетов")
    if 'raw_data' not in st.session_state:
        st.session_state.raw_data = None
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = pd.DataFrame()
    if 'ai_analysis' not in st.session_state:
        st.session_state.ai_analysis = {
            "columns": {},
            "metrics": [],
            "categories": [],
            "visualizations": []
        }
    st.markdown(
        """
        <style>
        .data-preview {
            border: 2px solid #2E86C1;
            border-radius: 10px;
            padding: 15px;
            margin: 15px 0;
            background-color: #f8f9fa;
        }
        .stSelectbox, .stButton {
            border: 2px solid #2E86C1 !important;
            border-radius: 8px !important;
            padding: 8px !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    upload_option = st.radio(
        "Выберите способ загрузки:",
        ["📝 Текстовый ввод", "📁 Файл"],
        horizontal=True
    )
    data_input = None
    if upload_option == "📝 Текстовый ввод":
        data_input = st.text_area(
            "Введите данные в формате CSV:",
            height=200,
            placeholder="Пример:\nДата,Категория,Сумма\n2024-01-01,Еда,1500"
        )
    else:
        uploaded_file = st.file_uploader(
            "Загрузите CSV или Excel файл",
            type=["csv", "xlsx"]
        )
        if uploaded_file:
            try:
                if uploaded_file.name.endswith(".csv"):
                    data_input = uploaded_file.getvalue().decode("utf-8")
                else:
                    df = pd.read_excel(uploaded_file)
                    data_input = df.to_csv(index=False)
            except Exception as e:
                st.error(f"Ошибка чтения файла: {str(e)}")
    if data_input and st.button("🔍 Анализировать данные"):
        with st.spinner("ИИ анализирует структуру..."):
            try:
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
                raw_response = LLMChain(
                    llm=llm,
                    prompt=PromptTemplate.from_template("{prompt}")
                ).run(prompt=analysis_prompt)
                json_str = re.search(r'\{.*\}', raw_response, re.DOTALL).group()
                analysis = json.loads(json_str)
                df = pd.read_csv(StringIO(data_input))
                valid_metrics = [col for col in analysis["metrics"] if col in df.columns]
                valid_categories = [col for col in analysis["categories"] if col in df.columns]
                st.session_state["processed_data"] = df
                st.session_state["ai_analysis"] = {
                    "metrics": valid_metrics or df.select_dtypes(include="number").columns.tolist(),
                    "categories": valid_categories or df.select_dtypes(exclude="number").columns.tolist(),
                    "visualizations": analysis["visualizations"]
                }
                st.success("Анализ завершен!")
            except json.JSONDecodeError:
                st.error("Ошибка формата ответа ИИ. Попробуйте другие данные.")
            except Exception as e:
                st.error(f"Критическая ошибка: {str(e)}")
    processed_data = st.session_state.get("processed_data", pd.DataFrame())
    ai_analysis = st.session_state.get("ai_analysis", {"visualizations": []})
    if not processed_data.empty:
        st.markdown("---")
        st.subheader("Визуализация данных")
        if not ai_analysis.get("visualizations"):
            st.info("Сначала выполните анализ данных для получения рекомендаций по визуализациям.")
            return
        viz_types = [v["type"] for v in ai_analysis["visualizations"]]
        selected_viz = st.selectbox("Тип визуализации", viz_types)
        viz_config = next(v for v in ai_analysis["visualizations"] if v["type"] == selected_viz)
        col1, col2 = st.columns(2)
        with col1:
            try:
                x_default = viz_config["x"] if viz_config["x"] in ai_analysis["categories"] else \
                ai_analysis["categories"][0]
                x_axis = st.selectbox("Ось X", ai_analysis["categories"],
                                      index=ai_analysis["categories"].index(x_default))
            except Exception:
                x_axis = st.selectbox("Ось X", ai_analysis["categories"])
        with col2:
            try:
                y_default = viz_config["y"] if viz_config["y"] in ai_analysis["metrics"] else ai_analysis["metrics"][0]
                y_axis = st.selectbox("Ось Y", ai_analysis["metrics"], index=ai_analysis["metrics"].index(y_default))
            except Exception:
                y_axis = st.selectbox("Ось Y", ai_analysis["metrics"])
        if st.button("Построить график"):
            try:
                if selected_viz == "line":
                    fig = px.line(processed_data, x=x_axis, y=y_axis)
                elif selected_viz == "bar":
                    fig = px.bar(processed_data, x=x_axis, y=y_axis)
                elif selected_viz == "pie":
                    fig = px.pie(processed_data, names=x_axis, values=y_axis)
                elif selected_viz == "scatter":
                    fig = px.scatter(processed_data, x=x_axis, y=y_axis)
                else:
                    st.error("Неподдерживаемый тип диаграммы.")
                    return
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Ошибка построения: {str(e)}")


def text_to_speech(text, lang='ru'):
    tts = gTTS(text=text, lang=lang)
    audio_bytes = io.BytesIO()
    tts.write_to_fp(audio_bytes)
    audio_bytes.seek(0)
    return audio_bytes


def play_audio(audio_bytes):
    audio_data, sample_rate = sf.read(audio_bytes)
    sd.play(audio_data, sample_rate)
    sd.wait()


def ai_assistant_page():
    st.header("💬 Чат с финансовым ассистентом")

    # Принудительно создаём новый экземпляр AudioRecorder,
    # чтобы гарантировать, что используется актуальный объект с методом to_wav_bytes.
    st.session_state.recorder = AudioRecorder()

    defaults = {
        'chat_history': [],
        'recording': False,
        'temp_audio': None,
        'show_preview': False,
        'recognized_text': ""
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    input_mode = st.radio("Режим ввода:", ["Текст", "Голос"], horizontal=True)

    # --- РЕЖИМ ГОЛОСА ---
    if input_mode == "Голос":
        st.markdown(
            """
            <style>
            button[aria-label="start"],
            button[aria-label="stop"],
            label[for="audio-input-select"] {
                visibility: hidden !important;
                height: 0 !important;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🎤 Начать запись", disabled=st.session_state.recording):
                st.session_state.recording = True
                st.session_state.recorder.recording = True
                st.session_state.show_preview = False
                st.session_state.temp_audio = None
                st.session_state.recognized_text = ""
        with col2:
            if st.button("⏹️ Остановить запись", disabled=not st.session_state.recording):
                st.session_state.recording = False
                st.session_state.recorder.recording = False
                audio_bytes = st.session_state.recorder.to_wav_bytes(sample_rate=16000)
                if audio_bytes:
                    st.session_state.temp_audio = audio_bytes
                    st.session_state.show_preview = True
                    recognized = audio_to_text(audio_bytes)
                    st.session_state.recognized_text = recognized
                st.session_state.recorder.frames = []
        if st.session_state.recording:
            st.info("🔴 Идёт запись голосового сообщения...")
        if st.session_state.show_preview and st.session_state.temp_audio:
            st.markdown("---")
            st.subheader("Предпросмотр записи")
            st.audio(st.session_state.temp_audio, format="audio/wav")
            st.write("**Расшифровка:**")
            st.text_area("Текст из голосового сообщения:", value=st.session_state.recognized_text, height=100)
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                if st.button("✅ Отправить голосовое"):
                    st.session_state.chat_history.append({
                        "role": "user",
                        "content": st.session_state.recognized_text,
                        "audio": st.session_state.temp_audio
                    })
                    st.session_state.show_preview = False
                    st.session_state.temp_audio = None
                    st.session_state.recognized_text = ""
                    safe_rerun()
            with col_b:
                if st.button("🔄 Перезаписать"):
                    st.session_state.show_preview = False
                    st.session_state.temp_audio = None
                    st.session_state.recognized_text = ""
            with col_c:
                if st.button("❌ Отменить"):
                    st.session_state.show_preview = False
                    st.session_state.temp_audio = None
                    st.session_state.recognized_text = ""
        webrtc_streamer(
            key="speech-to-text",
            mode=WebRtcMode.SENDONLY,
            audio_processor_factory=lambda: st.session_state.recorder,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={"audio": True},
        )
        user_input = ""
    else:
        user_input = st.text_input("Введите сообщение:")

    if 'audio_cache' not in st.session_state:
        st.session_state.audio_cache = {}

    st.markdown(
        """
        <style>
        .chat-message {
            padding: 1rem;
            border-radius: 1rem;
            margin: 0.5rem 0;
            max-width: 80%;
        }
        .user-message {
            background-color: #2E86C1;
            color: white;
            margin-left: auto;
        }
        .assistant-message {
            background-color: #f0f2f6;
            margin-right: auto;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-message user-message">{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-message assistant-message">{msg["content"]}</div>', unsafe_allow_html=True)
            if msg.get("audio"):
                st.audio(msg["audio"], format='audio/mp3')
    if input_mode == "Текст":
        col_send, _ = st.columns([1, 3])
        with col_send:
            if st.button("Отправить"):
                if user_input.strip():
                    st.session_state.chat_history.append({
                        "role": "user",
                        "content": user_input,
                        "audio": None
                    })
                    # Здесь можно добавить логику LLM, например LLMChain
    st.markdown("---")
    if st.button("Очистить историю"):
        st.session_state.chat_history = []
        safe_rerun()


def budget_tips_page():
    st.header("💡 Советы по планированию бюджета")
    st.subheader("📌 Основные принципы бюджетирования")
    st.write("""
    - **Правило 50/30/20:** 
        - 50% дохода на обязательные расходы (жилье, еда, транспорт).
        - 30% на желания (развлечения, хобби).
        - 20% на сбережения и инвестиции.
    - **Ведите учет расходов:** Записывайте все траты, чтобы понимать, куда уходят деньги.
    - **Планируйте заранее:** Составляйте бюджет на месяц вперед.
    """)
    st.subheader("📉 Советы по сокращению расходов")
    st.write("""
    - **Откажитесь от ненужных подписок:** Проверьте, какие подписки вы не используете.
    - **Покупайте с умом:** Используйте скидки, акции и сравнивайте цены.
    - **Готовьте дома:** Это дешевле и полезнее, чем питаться в кафе.
    - **Экономьте на коммунальных услугах:** Установите энергосберегающие приборы.
    """)
    st.subheader("📈 Советы по увеличению доходов")
    st.write("""
    - **Инвестируйте в себя:** Пройдите курсы, чтобы повысить квалификацию.
    - **Ищите дополнительные источники дохода:** Фриланс, подработка, инвестиции.
    - **Монетизируйте хобби:** Если вы хорошо рисуете, пишете или фотографируете, продавайте свои работы.
    """)
    st.subheader("📊 Инвестиционные советы")
    st.write("""
    - **Начните с малого:** Инвестируйте небольшие суммы, чтобы научиться.
    - **Диверсифицируйте портфель:** Вкладывайте в разные активы (акции, облигации, недвижимость).
    - **Долгосрочные инвестиции:** Не пытайтесь заработать быстро, думайте о долгосрочной перспективе.
    - **Используйте приложения для инвестиций:** Они помогут вам начать с минимальными знаниями.
    """)
    st.subheader("🧮 Интерактивный калькулятор бюджета")
    col1, col2 = st.columns(2)
    with col1:
        income = st.number_input("Ваш ежемесячный доход (руб.):", min_value=0)
    with col2:
        expenses = st.number_input("Ваши ежемесячные расходы (руб.):", min_value=0)
    if st.button("Рассчитать бюджет"):
        savings = income - expenses
        if savings > 0:
            st.success(f"✅ Вы можете отложить {savings} руб. в месяц!")
            st.write("Рекомендации:")
            if savings >= 0.2 * income:
                st.write("- Вы откладываете более 20% дохода. Это отличный результат!")
            else:
                st.write("- Попробуйте увеличить сбережения до 20% дохода.")
        else:
            st.error(f"❌ Вам нужно сократить расходы на {-savings} руб.")
            st.write("Рекомендации:")
            st.write("- Пересмотрите свои расходы и найдите категории, где можно сэкономить.")
    st.subheader("📋 Пример бюджета по правилу 50/30/20")
    if income > 0:
        necessities = 0.5 * income
        wants = 0.3 * income
        savings_investments = 0.2 * income
        st.write(f"""
        - **Обязательные расходы (50%):** {necessities:.2f} руб.
        - **Желания (30%):** {wants:.2f} руб.
        - **Сбережения и инвестиции (20%):** {savings_investments:.2f} руб.
        """)
    else:
        st.write("Введите ваш доход, чтобы увидеть пример бюджета.")


st.sidebar.title("🌐 Навигация")
page = st.sidebar.radio(
    "Выберите раздел:",
    ("Главная", "Мой профиль", "ИИ-ассистент", "Анализ моих отчетов", "Советы по планированию бюджета")
)

if page == "Главная":
    main_page()
elif page == "Мой профиль":
    profile_page()
elif page == "ИИ-ассистент":
    ai_assistant_page()
elif page == "Анализ моих отчетов":
    reports_page()
elif page == "Советы по планированию бюджета":
    budget_tips_page()


def process_file(uploaded_file):
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Поддерживаются только файлы CSV и XLSX.")
            return None
        st.session_state.financial_data = df
        st.success("Данные успешно загружены и сохранены!")
        return df.to_string(index=False)
    except Exception as e:
        st.error(f"Ошибка обработки файла: {e}")
        return None
