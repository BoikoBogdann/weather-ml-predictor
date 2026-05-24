import pandas as pd
import requests
import datetime

def fetch_historical_weather(lat: float, lon: float, start_date: str, end_date: str):
    """
    Офіційно верифікований скрапер: качає повний набір daily-параметрів 
    та автоматично перемикає сервери Архів/Прогноз.
    """
    today_str = datetime.date.today().strftime("%Y-%m-%d")
    
    if end_date >= today_str:
        base_url = "https://api.open-meteo.com/v1/forecast"
    else:
        base_url = "https://archive-api.open-meteo.com/v1/archive"
    
    # Строго ті назви, які підтримує сервер згідно з твоєю специфікацією
    daily_vars = [
        "precipitation_sum",           # Для таргету
        "temperature_2m_max",          # Залишаємо для загальної структури
        "temperature_2m_min",
        "temperature_2m_mean",
        "apparent_temperature_max",
        "apparent_temperature_min",
        "wind_speed_10m_max",
        "wind_gusts_10m_max",          # Пориви вітру
        "wind_direction_10m_dominant", # Напрямок вітру
        "shortwave_radiation_sum",     # Радіація
        "sunshine_duration",           # Тривалість сонця
        "daylight_duration",           # Тривалість дня
        "et0_fao_evapotranspiration"   # Випаровування
    ]
    
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": ",".join(daily_vars),
        "timezone": "auto"
    }
    
    try:
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if "daily" in data:
            return pd.DataFrame(data["daily"])
    except requests.exceptions.RequestException as e:
        print(f"Помилка запиту Open-Meteo: {e}")
    return None