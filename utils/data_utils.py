import pandas as pd
import numpy as np
import re
from typing import Optional, Dict, List, Any, Union
import json
from sklearn.linear_model import LinearRegression

def parse_financial_data(text: str) -> pd.DataFrame:
    """
    Parse arbitrary text into a structured DataFrame with financial data.
    
    Args:
        text: Raw text with financial information
        
    Returns:
        DataFrame with categories, types and amounts
    """
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

def parse_csv_data(uploaded_file) -> Optional[pd.DataFrame]:
    """
    Universal CSV file reader with error handling.
    
    Args:
        uploaded_file: File-like object with CSV data
        
    Returns:
        DataFrame or None if parsing failed
    """
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
        return None
    except pd.errors.ParserError:
        return None
    except Exception as e:
        print(f"Error parsing CSV: {str(e)}")
        return None

def analyze_financial_trends(data: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze financial data to identify trends and make projections.
    
    Args:
        data: DataFrame with financial data including date information
        
    Returns:
        Dictionary with trend analysis results
    """
    results = {"trends": [], "projections": {}, "anomalies": []}
    
    # Return early if no data or insufficient data
    if data.empty or len(data) < 3:
        return results
    
    # Check if we have time series data
    if "Дата" in data.columns and "Сумма" in data.columns:
        # Ensure date is in datetime format
        try:
            data["Дата"] = pd.to_datetime(data["Дата"])
            data = data.sort_values("Дата")
            
            # Analyze by category if available
            if "Категория" in data.columns:
                for category, group in data.groupby("Категория"):
                    if len(group) < 3:
                        continue
                        
                    # Calculate trend
                    X = np.array(range(len(group))).reshape(-1, 1)
                    y = group["Сумма"].values
                    model = LinearRegression()
                    model.fit(X, y)
                    
                    # Determine if increasing or decreasing
                    slope = model.coef_[0]
                    trend_direction = "увеличение" if slope > 0 else "уменьшение"
                    
                    # Calculate prediction for next period
                    next_x = np.array([[len(group)]])
                    prediction = model.predict(next_x)[0]
                    
                    results["trends"].append({
                        "category": category,
                        "direction": trend_direction,
                        "slope": float(slope),
                        "confidence": float(model.score(X, y))
                    })
                    
                    results["projections"][category] = float(prediction)
            
            # Detect anomalies (values more than 2 standard deviations from mean)
            if "Категория" in data.columns:
                for category, group in data.groupby("Категория"):
                    mean = group["Сумма"].mean()
                    std = group["Сумма"].std()
                    
                    if pd.notna(std) and std > 0:
                        anomalies = group[abs(group["Сумма"] - mean) > 2 * std]
                        
                        for _, row in anomalies.iterrows():
                            results["anomalies"].append({
                                "category": category,
                                "date": row["Дата"].strftime("%Y-%m-%d"),
                                "amount": float(row["Сумма"]),
                                "expected": float(mean),
                                "deviation": float((row["Сумма"] - mean) / std)
                            })
        except Exception as e:
            print(f"Error analyzing trends: {str(e)}")
    
    return results

def categorize_transactions(transactions: pd.DataFrame) -> pd.DataFrame:
    """
    Automatically categorize uncategorized transactions based on description patterns.
    
    Args:
        transactions: DataFrame with transaction data
        
    Returns:
        DataFrame with categorized transactions
    """
    if transactions.empty:
        return transactions
        
    # Create a copy to avoid modifying the original DataFrame
    result = transactions.copy()
    
    # Only proceed if we have description and amount columns
    if "Описание" in result.columns and "Сумма" in result.columns:
        # If we don't have a category column, add it
        if "Категория" not in result.columns:
            result["Категория"] = None
            
        # Define category patterns
        category_patterns = {
            "Продукты": ["магазин", "супермаркет", "продукт", "еда", "фрукт", "овощ", "молоко", "хлеб"],
            "Рестораны": ["ресторан", "кафе", "столовая", "фастфуд", "бар", "пицца", "суши"],
            "Транспорт": ["метро", "автобус", "такси", "uber", "яндекс.такси", "каршеринг", "бензин", "заправка"],
            "Жилье": ["аренда", "ком.услуги", "жкх", "интернет", "связь", "квартплата"],
            "Развлечения": ["кино", "театр", "концерт", "музей", "выставка", "парк", "отдых"],
            "Здоровье": ["аптека", "лекарств", "врач", "больница", "клиника", "медицин"],
            "Одежда": ["одежда", "обувь", "магазин", "zara", "h&m"],
            "Подписки": ["подписка", "netflix", "spotify", "youtube"]
        }
        
        # Iterate through uncategorized transactions
        for idx, row in result[result["Категория"].isnull()].iterrows():
            description = str(row["Описание"]).lower()
            
            # Find matching category
            for category, patterns in category_patterns.items():
                if any(pattern in description for pattern in patterns):
                    result.at[idx, "Категория"] = category
                    break
                    
            # If still uncategorized, mark as "Другое"
            if pd.isnull(result.at[idx, "Категория"]):
                result.at[idx, "Категория"] = "Другое"
    
    return result
