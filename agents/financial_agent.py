from typing import Dict, List, Any, Optional
import pandas as pd
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import json
import re

class FinancialAgent:
    """
    A logical agent that handles complex financial queries such as budget planning,
    financial report analysis, or comparison of investment options.
    """
    def __init__(self, llm):
        self.llm = llm
        
    def handle_complex_query(self, query: str, financial_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Processes complex financial queries and returns structured information.
        
        Args:
            query: The user's complex financial query
            financial_data: DataFrame containing the user's financial data
            
        Returns:
            A dictionary containing the query results, recommendations, and any visualizations needed
        """
        data_description = self._generate_data_description(financial_data)
        
        # Create a task-specific prompt
        prompt = f"""
        Ты профессиональный финансовый аналитик, которому нужно решить сложную финансовую задачу пользователя.
        
        Данные пользователя:
        {data_description}
        
        Запрос пользователя: {query}
        
        Выполни следующие шаги:
        1. Определи тип финансовой задачи (бюджетирование, анализ расходов, планирование инвестиций и т.д.)
        2. Проанализируй предоставленные данные в контексте запроса
        3. Сформируй детальный ответ, включая:
           - Анализ текущей ситуации
           - Конкретные рекомендации
           - Предложения по визуализации данных (если применимо)
        
        Верни результат в формате JSON со следующей структурой:
        {{
            "task_type": "тип задачи",
            "analysis": "подробный анализ ситуации",
            "recommendations": ["список конкретных рекомендаций"],
            "visualizations": [
                {{
                    "type": "тип визуализации (pie, bar, line и т.д.)",
                    "title": "название графика",
                    "description": "описание графика",
                    "data": {{
                        "x": "название столбца для оси X или категорий",
                        "y": "название столбца для значений"
                    }}
                }}
            ]
        }}
        """
        
        # Generate response using LLM
        raw_response = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate.from_template("{prompt}")
        ).run(prompt=prompt)
        
        # Extract JSON from response
        try:
            json_str = re.search(r'\{.*\}', raw_response, re.DOTALL).group()
            response_data = json.loads(json_str)
        except (AttributeError, json.JSONDecodeError):
            # Fallback if JSON parsing fails
            response_data = {
                "task_type": "Анализ запроса",
                "analysis": "Не удалось структурировать ответ в JSON формате. Вот исходный ответ:",
                "recommendations": ["Попробуйте уточнить ваш запрос"],
                "visualizations": [],
                "raw_response": raw_response
            }
            
        return response_data
    
    def create_budget_plan(self, financial_data: pd.DataFrame, period: str = "месяц") -> Dict[str, Any]:
        """
        Creates a budget plan based on historical financial data.
        
        Args:
            financial_data: DataFrame containing the user's financial data
            period: The period for which to create the budget (month, quarter, year)
            
        Returns:
            A dictionary containing the budget plan
        """
        data_description = self._generate_data_description(financial_data)
        
        prompt = f"""
        Ты финансовый планировщик, которому нужно создать бюджетный план на {period} для пользователя.
        
        Данные пользователя:
        {data_description}
        
        Создай детальный бюджетный план, учитывая:
        1. Исторические расходы по категориям
        2. Доходы пользователя
        3. Баланс между необходимыми и дискреционными расходами
        4. Возможности для сбережений и инвестиций
        
        Верни результат в формате JSON со следующей структурой:
        {{
            "income": {{
                "categories": ["категории доходов"],
                "amounts": [суммы]
            }},
            "expenses": {{
                "categories": ["категории расходов"],
                "amounts": [суммы],
                "is_necessary": [true/false для каждой категории]
            }},
            "savings": число,
            "recommendations": ["список рекомендаций по улучшению бюджета"],
            "summary": "краткое описание бюджетного плана"
        }}
        """
        
        # Generate response using LLM
        raw_response = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate.from_template("{prompt}")
        ).run(prompt=prompt)
        
        # Extract JSON from response
        try:
            json_str = re.search(r'\{.*\}', raw_response, re.DOTALL).group()
            budget_plan = json.loads(json_str)
        except (AttributeError, json.JSONDecodeError):
            # Fallback if JSON parsing fails
            budget_plan = {
                "income": {"categories": [], "amounts": []},
                "expenses": {"categories": [], "amounts": [], "is_necessary": []},
                "savings": 0,
                "recommendations": ["Не удалось создать детальный бюджетный план из-за ошибки обработки данных."],
                "summary": "Произошла ошибка при создании бюджетного плана. Попробуйте предоставить более подробные данные о своих финансах."
            }
            
        return budget_plan
        
    def analyze_investment_options(self, query: str, financial_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyzes investment options based on user query and financial situation.
        
        Args:
            query: The user's investment query
            financial_data: DataFrame containing the user's financial data
            
        Returns:
            A dictionary containing investment analysis and recommendations
        """
        data_description = self._generate_data_description(financial_data)
        
        prompt = f"""
        Ты финансовый советник, специализирующийся на инвестициях. Пользователь просит совета по инвестициям.
        
        Данные пользователя:
        {data_description}
        
        Запрос пользователя: {query}
        
        Проанализируй финансовую ситуацию пользователя и предложи варианты инвестирования, учитывая:
        1. Текущее финансовое положение
        2. Долгосрочные и краткосрочные цели
        3. Толерантность к риску (определи на основе запроса)
        4. Диверсификацию портфеля
        
        Верни результат в формате JSON со следующей структурой:
        {{
            "financial_assessment": "оценка текущего финансового положения",
            "risk_profile": "предполагаемый профиль риска пользователя",
            "investment_options": [
                {{
                    "name": "название инвестиционного инструмента",
                    "type": "тип (акции, облигации, ETF и т.д.)",
                    "risk_level": "уровень риска (низкий, средний, высокий)",
                    "expected_return": "ожидаемая доходность в %",
                    "description": "краткое описание"
                }}
            ],
            "strategy": "рекомендуемая инвестиционная стратегия",
            "allocation": {{
                "categories": ["категории инвестиций"],
                "percentages": [проценты распределения]
            }}
        }}
        """
        
        # Generate response using LLM
        raw_response = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate.from_template("{prompt}")
        ).run(prompt=prompt)
        
        # Extract JSON from response
        try:
            json_str = re.search(r'\{.*\}', raw_response, re.DOTALL).group()
            investment_analysis = json.loads(json_str)
        except (AttributeError, json.JSONDecodeError):
            # Fallback if JSON parsing fails
            investment_analysis = {
                "financial_assessment": "Не удалось провести оценку из-за ошибки обработки данных.",
                "risk_profile": "Не определено",
                "investment_options": [],
                "strategy": "Для получения точного инвестиционного анализа, пожалуйста, уточните ваш запрос и предоставьте более детальную информацию.",
                "allocation": {"categories": [], "percentages": []}
            }
            
        return investment_analysis
        
    def _generate_data_description(self, financial_data: pd.DataFrame) -> str:
        """Generates a textual description of the financial data"""
        if financial_data.empty:
            return "Нет доступных финансовых данных."
            
        description = []
        
        # Check if we have income and expense data
        if "Тип" in financial_data.columns:
            income_data = financial_data[financial_data["Тип"] == "Доходы"]
            expense_data = financial_data[financial_data["Тип"] == "Расходы"]
            
            total_income = income_data["Сумма"].sum() if not income_data.empty else 0
            total_expenses = expense_data["Сумма"].sum() if not expense_data.empty else 0
            
            description.append(f"Общий доход: {total_income} руб.")
            description.append(f"Общие расходы: {total_expenses} руб.")
            description.append(f"Баланс: {total_income - total_expenses} руб.")
            
            # Add category breakdown
            if "Категория" in financial_data.columns:
                description.append("\nДоходы по категориям:")
                if not income_data.empty:
                    for category, group in income_data.groupby("Категория"):
                        cat_sum = group["Сумма"].sum()
                        description.append(f"- {category}: {cat_sum} руб. ({cat_sum/total_income*100:.1f}% от общего дохода)")
                else:
                    description.append("- Нет данных о доходах")
                    
                description.append("\nРасходы по категориям:")
                if not expense_data.empty:
                    for category, group in expense_data.groupby("Категория"):
                        cat_sum = group["Сумма"].sum()
                        description.append(f"- {category}: {cat_sum} руб. ({cat_sum/total_expenses*100:.1f}% от общих расходов)")
                else:
                    description.append("- Нет данных о расходах")
        else:
            # Simple description without income/expense classification
            if "Категория" in financial_data.columns and "Сумма" in financial_data.columns:
                description.append("Финансовые данные по категориям:")
                for category, group in financial_data.groupby("Категория"):
                    cat_sum = group["Сумма"].sum()
                    description.append(f"- {category}: {cat_sum} руб.")
            else:
                description.append("Доступны финансовые данные, но без категоризации.")
                
        return "\n".join(description)
