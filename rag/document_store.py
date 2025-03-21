import os
import json
import pandas as pd
from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class DocumentStore:
    """
    Simple document store to manage financial knowledge base documents
    and user financial data for retrieval.
    """
    def __init__(self):
        self.documents = []
        self.finance_knowledge = self._load_finance_knowledge()
        self.vectorizer = TfidfVectorizer()
        self.document_vectors = None
        
    def _load_finance_knowledge(self) -> List[Dict[str, str]]:
        """Load predefined financial knowledge"""
        knowledge = [
            {
                "id": "budget_planning",
                "title": "Бюджетное планирование",
                "content": "Бюджетное планирование - это процесс организации расходов и доходов на определенный период. "
                          "Рекомендуется использовать правило 50/30/20: 50% на необходимые расходы, 30% на желания, "
                          "20% на сбережения и инвестиции."
            },
            {
                "id": "emergency_fund",
                "title": "Аварийный фонд",
                "content": "Аварийный фонд должен составлять 3-6 месячных расходов. Храните его в ликвидных активах "
                          "с низким риском, например, на сберегательном счете."
            },
            {
                "id": "debt_management",
                "title": "Управление долгами",
                "content": "Существует две стратегии погашения долгов: 'снежный ком' (сначала погашаем долги с наименьшей суммой) "
                          "и 'лавина' (сначала погашаем долги с наибольшей процентной ставкой)."
            },
            {
                "id": "investment_basics",
                "title": "Основы инвестирования",
                "content": "Диверсификация снижает риск. Распределите инвестиции между акциями, облигациями, недвижимостью и другими активами. "
                          "Для долгосрочных инвестиций рассмотрите индексные фонды с низкими комиссиями."
            },
            {
                "id": "saving_strategies",
                "title": "Стратегии накопления",
                "content": "Автоматизируйте сбережения - настройте автоматический перевод части зарплаты на сберегательный счет. "
                          "Старайтесь сберегать не менее 10-15% ежемесячного дохода."
            },
            {
                "id": "retirement_planning",
                "title": "Планирование пенсии",
                "content": "Начинайте планировать пенсию как можно раньше. Используйте пенсионные фонды и инвестиционные инструменты "
                          "с налоговыми льготами."
            }
        ]
        return knowledge
    
    def add_financial_data(self, df: pd.DataFrame):
        """Add financial data to the document store"""
        # Transform DataFrame into documents
        if not df.empty:
            # Add overall summary of the financial data
            total_income = df[df["Тип"] == "Доходы"]["Сумма"].sum() if "Тип" in df.columns else 0
            total_expenses = df[df["Тип"] == "Расходы"]["Сумма"].sum() if "Тип" in df.columns else 0
            
            # Add summary document
            summary_doc = {
                "id": "financial_summary",
                "title": "Сводка по финансовым данным",
                "content": f"Общий доход: {total_income} руб. Общие расходы: {total_expenses} руб. "
                          f"Баланс: {total_income - total_expenses} руб."
            }
            self.documents.append(summary_doc)
            
            # Add category breakdowns if available
            if "Категория" in df.columns and "Сумма" in df.columns:
                for category in df["Категория"].unique():
                    category_data = df[df["Категория"] == category]
                    category_total = category_data["Сумма"].sum()
                    category_type = category_data["Тип"].iloc[0] if "Тип" in df.columns else "Не указано"
                    
                    category_doc = {
                        "id": f"category_{category.lower().replace(' ', '_')}",
                        "title": f"Категория: {category}",
                        "content": f"Категория {category} ({category_type}): общая сумма {category_total} руб."
                    }
                    self.documents.append(category_doc)
            
            # Vectorize the documents
            self._vectorize_documents()
    
    def add_document(self, document: Dict[str, str]):
        """Add single document to the store"""
        self.documents.append(document)
        # Re-vectorize documents
        self._vectorize_documents()
    
    def _vectorize_documents(self):
        """Create vector representations of documents"""
        if not self.documents:
            return
            
        # Combine finance knowledge and user documents
        all_docs = self.finance_knowledge + self.documents
        
        # Extract contents for vectorization
        contents = [doc["content"] for doc in all_docs]
        
        # Fit vectorizer and transform documents
        self.document_vectors = self.vectorizer.fit_transform(contents)
        
    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Search for documents relevant to the query"""
        if not self.documents and not self.finance_knowledge:
            return []
            
        # Combine finance knowledge and user documents
        all_docs = self.finance_knowledge + self.documents
        
        if self.document_vectors is None:
            self._vectorize_documents()
            
        # Vectorize the query
        query_vector = self.vectorizer.transform([query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_vector, self.document_vectors).flatten()
        
        # Get top-k document indices
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        # Return top-k documents with similarity scores
        return [
            {**all_docs[idx], "similarity": float(similarities[idx])}
            for idx in top_indices
        ]
