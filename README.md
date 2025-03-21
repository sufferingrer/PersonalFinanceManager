# Финансовый ассистент с RAG и голосовым интерфейсом

Интеллектуальная система для персонализированного финансового анализа и рекомендаций, использующая технологии Retrieval-Augmented Generation (RAG) и специализированных финансовых агентов.

## Возможности

- **Анализ финансовых данных** - загрузка, обработка и визуализация финансовой информации
- **Голосовой и текстовый интерфейс** - взаимодействие через ввод текста или голосовые команды
- **RAG система** - генерация ответов на основе релевантного контекста из базы знаний
- **Специализированные финансовые агенты** - решение конкретных финансовых задач с учетом данных пользователя
- **Интерактивное планирование бюджета** - создание оптимального финансового плана

## Технологии

- **Streamlit** - веб-интерфейс и визуализация
- **DeepSeek API** - продвинутая языковая модель для генерации ответов
- **LangChain** - фреймворк для работы с языковыми моделями и RAG
- **streamlit-webrtc** - голосовой интерфейс через браузер
- **SpeechRecognition** - распознавание речи
- **gTTS** - синтез речи
- **Pandas, Plotly** - обработка и визуализация данных

## Архитектура

Приложение имеет модульную архитектуру, состоящую из нескольких ключевых компонентов:

### RAG (Retrieval-Augmented Generation)

RAG система обеспечивает точные и информативные ответы путем обогащения языковой модели релевантным контекстом:

1. **DocumentStore** - хранилище финансовых знаний и данных пользователя
2. **RAGRetriever** - извлечение релевантных документов и генерация ответов

### Финансовые агенты

Специализированные модули для решения конкретных финансовых задач:

1. **FinancialAgent** - логический агент для обработки сложных финансовых запросов
2. **Планирование бюджета** - анализ расходов и создание оптимального бюджета
3. **Инвестиционные рекомендации** - анализ инвестиционных возможностей

### Голосовой интерфейс

Компоненты для обработки аудио и обеспечения голосового взаимодействия:

1. **AudioRecorder** - запись аудио через WebRTC
2. **AudioProcessor** - преобразование текста в речь и речи в текст

## Установка и запуск

### Требования

- Python 3.8+
- Зависимости из файла requirements.txt
- DeepSeek API ключ

### Установка

```bash
# Установка зависимостей
pip install -r requirements.txt
```

### Настройка API ключей

Для работы с DeepSeek API требуется ключ доступа. Создайте файл `.env` в корне проекта со следующим содержимым:

```
DEEPSEEK_API_KEY=ваш_ключ_deepseek_api
```

Или установите переменную окружения:

```bash
export DEEPSEEK_API_KEY=ваш_ключ_deepseek_api
```

### Запуск приложения

```bash
streamlit run app.py
```

## Использование

1. **Загрузка данных** - загрузите финансовые данные в CSV формате или введите их вручную
2. **Анализ данных** - просмотрите автоматический анализ и визуализации
3. **Финансовый помощник** - задавайте вопросы текстом или голосом для получения рекомендаций
4. **Планирование бюджета** - создавайте и настраивайте личный бюджетный план

## Структура проекта

```
.
├── app.py                  # Основной файл приложения
├── rag/                    # Модули для RAG системы
│   ├── document_store.py   # Хранилище документов
│   └── retriever.py        # Извлечение документов и генерация ответов
├── agents/                 # Финансовые агенты
│   └── financial_agent.py  # Основной класс финансового агента
├── utils/                  # Вспомогательные утилиты
│   ├── audio_utils.py      # Обработка аудио
│   ├── data_utils.py       # Обработка данных
│   └── nlp_utils.py        # Обработка естественного языка
```

## Лицензия

MIT