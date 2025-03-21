import re
from typing import Dict, List, Any, Optional, Tuple, Union

def is_complex_query(query: str) -> bool:
    """
    Determine if a query is complex (requires advanced financial analysis).
    
    Args:
        query: User query text
        
    Returns:
        Boolean indicating if query is complex
    """
    complex_keywords = [
        "бюджет", "планирование", "анализ", "сравнение", "инвестиции", 
        "прогноз", "оптимизация", "стратегия", "риск", "портфель",
        "рекомендации", "альтернативы", "диверсификация", "доходность"
    ]
    return any(keyword in query.lower() for keyword in complex_keywords)

def handle_unclear_request(query: str) -> Optional[str]:
    """
    Detect unclear requests and generate appropriate responses.
    
    Args:
        query: User query text
        
    Returns:
        Response message for unclear requests or None if request is clear
    """
    unclear_phrases = [
        "не знаю", "пример", "помощь", "подскажи", "как", "что", "непонятно",
        "непонятный", "способы", "варианты", "объясни", "расскажи"
    ]
    
    # Check if query is too short
    if len(query.split()) < 3:
        return "Ваш запрос слишком короткий. Пожалуйста, опишите подробнее, что вы хотели бы узнать о ваших финансах."
    
    # Check for unclear phrases
    if any(phrase in query.lower() for phrase in unclear_phrases):
        return """
        Пожалуйста, уточните ваш запрос. Вот несколько примеров того, что вы можете спросить:
        - "Какие у меня расходы по категориям за последний месяц?"
        - "Помоги составить бюджет на следующий месяц"
        - "Проанализируй мои финансы и предложи, как сэкономить"
        - "Где я трачу больше всего денег?"
        - "Составь инвестиционный план на 5 лет"
        """
    
    return None

def extract_date_range(query: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract date range information from a query.
    
    Args:
        query: User query text
        
    Returns:
        Tuple of (start_date, end_date) as strings or None if not found
    """
    # Extract date patterns like "с 01.01.2023 по 31.01.2023" or "за январь 2023"
    date_pattern = r"с\s+(\d{1,2}[./-]\d{1,2}[./-]\d{2,4})\s+по\s+(\d{1,2}[./-]\d{1,2}[./-]\d{2,4})"
    period_pattern = r"за\s+(январь|февраль|март|апрель|май|июнь|июль|август|сентябрь|октябрь|ноябрь|декабрь)(?:\s+(\d{4}))?"
    
    # Try to match date range pattern
    date_match = re.search(date_pattern, query, re.IGNORECASE)
    if date_match:
        return date_match.group(1), date_match.group(2)
    
    # Try to match period pattern
    period_match = re.search(period_pattern, query, re.IGNORECASE)
    if period_match:
        month = period_match.group(1)
        year = period_match.group(2) if period_match.group(2) else "2023"  # Default to current year if not specified
        
        # Map month name to number
        month_map = {
            "январь": "01", "февраль": "02", "март": "03", "апрель": "04",
            "май": "05", "июнь": "06", "июль": "07", "август": "08",
            "сентябрь": "09", "октябрь": "10", "ноябрь": "11", "декабрь": "12"
        }
        
        month_num = month_map.get(month.lower())
        if month_num:
            start_date = f"01.{month_num}.{year}"
            
            # Calculate end date based on month
            days_in_month = {
                "01": "31", "02": "28", "03": "31", "04": "30", "05": "31", "06": "30",
                "07": "31", "08": "31", "09": "30", "10": "31", "11": "30", "12": "31"
            }
            # Handle leap years for February
            if month_num == "02" and int(year) % 4 == 0 and (int(year) % 100 != 0 or int(year) % 400 == 0):
                days_in_month["02"] = "29"
                
            end_date = f"{days_in_month[month_num]}.{month_num}.{year}"
            return start_date, end_date
    
    return None, None

def extract_financial_goal(query: str) -> Optional[Dict[str, Any]]:
    """
    Extract financial goal information from a query.
    
    Args:
        query: User query text
        
    Returns:
        Dictionary with goal information or None if not found
    """
    # Match patterns like "накопить 100000 рублей за 6 месяцев на отпуск"
    goal_pattern = r"накопить\s+(\d+(?:\s*\d+)*)\s*(?:руб(?:лей)?|₽)(?:\s+за\s+(\d+)\s+(дн(?:я|ей)|недел(?:ю|и|ь)|месяц(?:а|ев)?|год(?:а)?))?\s+(?:на|для)\s+(.+?)(?:\.|$)"
    
    goal_match = re.search(goal_pattern, query, re.IGNORECASE)
    if goal_match:
        # Extract and clean up the amount
        amount_str = re.sub(r'\s+', '', goal_match.group(1))
        amount = int(amount_str)
        
        # Extract time period if specified
        period_value = int(goal_match.group(2)) if goal_match.group(2) else None
        period_unit = goal_match.group(3) if goal_match.group(3) else None
        
        # Normalize period unit
        if period_unit:
            if period_unit.startswith("дн"):
                period_unit = "days"
            elif period_unit.startswith("недел"):
                period_unit = "weeks"
            elif period_unit.startswith("месяц"):
                period_unit = "months"
            elif period_unit.startswith("год"):
                period_unit = "years"
        
        # Extract goal purpose
        purpose = goal_match.group(4).strip()
        
        return {
            "type": "savings",
            "amount": amount,
            "period_value": period_value,
            "period_unit": period_unit,
            "purpose": purpose
        }
    
    return None

def extract_category_info(query: str) -> List[str]:
    """
    Extract category information from a query.
    
    Args:
        query: User query text
        
    Returns:
        List of categories mentioned in the query
    """
    common_categories = [
        "продукты", "транспорт", "жилье", "развлечения", "здоровье", 
        "образование", "одежда", "рестораны", "путешествия", "связь",
        "подписки", "коммунальные", "спорт", "хобби", "подарки"
    ]
    
    found_categories = []
    for category in common_categories:
        if category in query.lower():
            found_categories.append(category)
            
    return found_categories
