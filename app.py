import streamlit as st
import folium
import os
import datetime
import requests
import pandas as pd
import plotly.express as px
from streamlit_folium import st_folium

# Імпорт нашого ML-ядра та скрапера
from data_scrapper import fetch_historical_weather
from pipeline import prepare_data, train_and_compare_models, make_prediction

st.set_page_config(page_title="Weather Predictor", layout="wide")

st.title("🌧️ Прототип сервісу прогнозування опадів")

# ==============================================================================
# 🧠 ІНІЦІАЛІЗАЦІЯ ПАМ'ЯТІ СЕСІЇ (SESSION STATE)
# ==============================================================================
if "df_weather" not in st.session_state: st.session_state.df_weather = None
if "best_model_name" not in st.session_state: st.session_state.best_model_name = None
if "feature_df" not in st.session_state: st.session_state.feature_df = None
if "rf_metrics" not in st.session_state: st.session_state.rf_metrics = None
if "lr_metrics" not in st.session_state: st.session_state.lr_metrics = None
if "gb_metrics" not in st.session_state: st.session_state.gb_metrics = None
if "best_params" not in st.session_state: st.session_state.best_params = None
if "trained_model_object" not in st.session_state: st.session_state.trained_model_object = None
if "df_forecast_results" not in st.session_state: st.session_state.df_forecast_results = None
if "forecast_start_date" not in st.session_state: st.session_state.forecast_start_date = None
if "forecast_end_date" not in st.session_state: st.session_state.forecast_end_date = None

# НАУКОВО-МЕТОДИЧНИЙ ДОВІДНИК
with st.expander("📘 Переглянути розширений науково-методичний довідник проєкту"):
    st.markdown("""
    ### 🧠 Фізичне обґрунтування та відбір ознак (Feature Selection)
    Для підвищення узагальнюючої здатності моделей та усунення ефекту **мультиколінеарності** (інформаційного дублювання) було проведено аналітичну фільтрацію простору ознак. Замість лінійно залежних температурних екстремумів (мінімальних та максимальних значень), у систему інтегровано **розширений «золотий набір» із 10 незалежних синоптичних та часових ознак**. 
    
    Кожен параметр відображає конкретний термодинамічний або гідродинамічний процес в атмосфері, необхідний для утворення хмарності та випадіння конденсату:
    
    1. **Сумарна сонячна радіація (`shortwave_radiation_sum`, МДж/м²):** Визначає енергетичний баланс атмосфери. Низькі значення вдень є прямим індикатором суцільної фронтальної хмарності, що передує опадам.
    2. **Тривалість сонячного сяйва (`sunshine_duration`, секунди):** Прямий синоптичний маркер. Що менше секунд чисте сонце пробивається до землі через хмари, то вища щільність хмарового покриву nimbus/cumulus і то вища ймовірність дощу.
    3. **Тривалість світлового дня (`daylight_duration`, секунди):** Астрономічний показник, який дозволяє моделі точно ідентифікувати сезонні кліматичні тренди (зима/літо) без використання громіздких категоріальних змінних.
    4. **Відчувальна максимальна температура (`apparent_temperature_max`, °C):** Комплексний біокліматичний індекс, що сигналізує про «паркість» повітря та тепловий потенціал атмосфери перед конвективними грозами.
    5. **Середня температура повітря (`temperature_2m_mean`, °C):** Відображає загальний термодинамічний стан повітряної маси на висоті 2 метрів, визначаючи її вологомісткість та фазовий стан опадів (дощ чи сніг).
    6. **Максимальна швидкість вітру (`wind_speed_10m_max`, км/год):** Динамічний маркер баричних градієнтів, що фіксує переміщення повітряних мас.
    7. **Пориви вітру (`wind_gusts_10m_max`, км/год):** Ключовий індикатор проходження контрастних атмосферних фронтів та формування шквалистих грозових осередків.
    8. **Домінантний напрямок вітру (`wind_direction_10m_dominant`, градуси):** Показує вектор руху повітряних мас, дозволяючи моделі оцінити, звідки заходить фронт — з вологих морських чи сухих континентальних басейнів.
    9. **Сумарне потенційне випаровування (`et0_fao_evapotranspiration`, мм):** Розраховується за стандартом ФАО. Показує об'єм вологи, що піднявся в атмосферу внаслідок транспірації, формуючи локальний потенціал для майбутньої конденсації.
    10. **Порядковий день року (`dayofyear`, 1–365):** Математична безперервна ознака для точного моделювання сезонної циклічності та кліматичних хвиль конкретного регіону.
    
    ### 🛡️ Захист від витоку даних (Data Leakage Protection)
    Згідно з методичними вимогами розробки інтелектуальних систем, параметри прямих опадів (`precipitation_sum`, `rain_sum`, `snowfall_sum`) **повністю виключено з матриці ознак $X$**. Стовпчик суми опадів використовується виключно на етапі передобробки для формування бінарного вектора відповідей (`target = 1` або `0`). Це гарантує чесність навчання моделі та унеможливлює ситуацію, коли алгоритм «підглядає» у фінальну відповідь під час історичного тестування.
    
    ### 📏 Методичні вказівки щодо вибору порогового значення опадів
    Бінарне розділення цільової змінної виконується динамічно за допомогою слайдера на основі порогів Всесвітньої метеорологічної організації (ВМО):
    * **0.1 мм — Глобальний мінімум (Морський клімат):** Фіксує навіть слабку мряку чи туманні опади. Використовується для регіонів з високою базовою вологості.
    * **0.5 мм — Континентальний стандарт (Рекомендовано для України):** Оптимальний аналітичний поріг. Очищує вибірку від випадкових інструментальних шумів (мікро-роса на датчиках станції, пориви вітру з вологою) та дрібного дощу, який повністю випаровується в повітрі, не досягаючи поверхні землі.
    * **1.0 мм — Прагматичний мінімум (Аридний/Сухий клімат):** Фокусує моделі виключно на значних зливах, ігноруючи слабкі опади, які моментально поглинаються пересушеною атмосферою.
    """)

@st.cache_data(ttl=3600)
def get_location_by_ip():
    try:
        response = requests.get("http://ip-api.com/json/", timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data["status"] == "success":
                return data["lat"], data["lon"], data["city"]
    except: pass
    return 48.467, 35.040, "Дніпро (за замовчуванням)"

start_lat, start_lon, current_city = get_location_by_ip()
st.write(f"🌍 Ваша поточна локація (визначено за IP): **{current_city}**")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🗺️ 1. Оберіть точку на карті:")
    m = folium.Map(location=[start_lat, start_lon], zoom_start=10)
    m.add_child(folium.LatLngPopup())
    map_data = st_folium(m, width=800, height=500, key="map")

with col2:
    st.subheader("⚙️ 2. Панель керування та налаштувань:")
    
    # ТАБИ для налаштування вхідних параметрів
    tab_hist_inputs, tab_pred_inputs = st.tabs(["📊 Налаштування Історії", "🔮 Налаштування Прогнозу"])
    
    today = datetime.date.today()
    
    with tab_hist_inputs:
        start_date = st.date_input("Дата початку історії (для навчання):", today - datetime.timedelta(days=3*365))
        end_date = st.date_input("Дата кінця історії (для навчання):", today - datetime.timedelta(days=1))
        threshold = st.slider("📏 Поріг відсікання опадів (в мм):", min_value=0.1, max_value=1.0, value=0.5, step=0.1)
        
    with tab_pred_inputs:
        st.write("Оберіть часове вікно майбутнього прогнозу:")
        pred_start = st.date_input("Початок прогнозу:", today)
        pred_end = st.date_input("Кінець прогнозу:", today + datetime.timedelta(days=7))
    
    # Координати точки
    if map_data and map_data.get("last_clicked"):
        lat = map_data["last_clicked"]["lat"]
        lon = map_data["last_clicked"]["lng"]
        st.success(f"📍 Локація обрана: {lat:.4f}, {lon:.4f}")
    else:
        lat = start_lat
        lon = start_lon
        st.info(f"📍 Точка за замовчуванням: {lat:.4f}, {lon:.4f}")
        
    # Головна кнопка
    if st.button("🚀 Запустити повний ML-Пайплайн", type="primary", use_container_width=True):
        st.session_state.df_forecast_results = None 
        st.session_state.forecast_start_date = pred_start
        st.session_state.forecast_end_date = pred_end
        
        forecast_days = (pred_end - pred_start).days + 1
        
        if forecast_days <= 0:
            st.error("❌ Дата кінця прогнозу не може бути ранішою за дату початку.")
        elif forecast_days > 31:
            st.error(f"❌ Обраний період прогнозу занадто великий ({forecast_days} днів). Максимальна тривалість обмежена 31 днем.")
        else:
            with st.spinner("⏳ Крок 1: Завантажуємо історичну базу для навчання..."):
                df_res = fetch_historical_weather(lat, lon, str(start_date), str(end_date))
                
            if df_res is not None and not df_res.empty:
                if len(df_res) < 180:
                    st.error(f"❌ Обраний історичний період занадто малий. Завантажте хоча б 180 днів.")
                else:
                    st.session_state.df_weather = df_res
                    
                    with st.spinner("🤖 Крок 2: Навчаємо та порівнюємо 3 моделі..."):
                        X, y = prepare_data(st.session_state.df_weather, threshold)
                        res = train_and_compare_models(X, y)
                        st.session_state.best_model_name = res[0]
                        st.session_state.feature_df = res[1]
                        st.session_state.rf_metrics = res[2]
                        st.session_state.lr_metrics = res[3]
                        st.session_state.gb_metrics = res[4]
                        st.session_state.best_params = res[5]
                        
                        from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
                        if "Random Forest" in res[0]:
                            model_obj = RandomForestClassifier(n_estimators=100, max_depth=res[5].get("max_depth"), random_state=42, class_weight='balanced', n_jobs=-1)
                        else:
                            model_obj = HistGradientBoostingClassifier(max_depth=res[5].get("max_depth"), random_state=42)
                            
                        model_obj.fit(X, y)
                        st.session_state.trained_model_object = model_obj
                        
                        os.makedirs("data", exist_ok=True)
                        df_res.to_csv("data/weather_daily.csv", index=False)
                    
                    # --- КРОК 3: ГЕНЕРАЦІЯ ПРОГНОЗУ НА МАЙБУТНЄ ---
                    with st.spinner("📡 Крок 3: Завантажуємо прогнозні фічі та запускаємо інференс..."):
                        df_future = fetch_historical_weather(lat, lon, str(pred_start), str(pred_end))
                        
                        if df_future is not None and not df_future.empty:
                            X_future, _ = prepare_data(df_future, threshold)
                            preds, probs = make_prediction(st.session_state.trained_model_object, X_future)
                            
                            st.session_state.df_forecast_results = pd.DataFrame({
                                "Дата": pd.to_datetime(df_future['time']).dt.strftime('%Y-%m-%d'),
                                "Середня температура (°C)": df_future['temperature_2m_mean'],
                                "Швидкість вітру (км/год)": df_future['wind_speed_10m_max'],
                                "Ймовірність опадів (%)": probs,
                                "Вердикт ML-ядра": ["🌧️ Опади" if p == 1 else "☀️ Сухо" for p in preds]
                            })
                        else:
                            st.error("❌ Не вдалося отримати прогнозні метеодані.")
            else:
                st.error("❌ Не вдалося отримати історичні дані з API.")

# ==============================================================================
# 📊 ГОЛОВНИЙ БЛОК ВІДОБРАЖЕННЯ РЕЗУЛЬТАТІВ (РОЗДІЛЕННЯ НА ВКЛАДКИ)
# ==============================================================================
if st.session_state.df_weather is not None:
    st.markdown("---")
    st.success("🎉 Пайплайн розрахунків виконано успішно!")
    
    # РОЗДІЛЯЄМО РЕЗУЛЬТАТ НА ОКРЕМІ КРУПНІ ВКЛАДКИ НА ЕКРАНІ
    view_tab_pred, view_tab_hist = st.tabs(["🔮 КЛІМАТИЧНИЙ ПРОГНОЗ (МАЙБУТНЄ)", "📊 АНАЛІТИКА ТА НАВЧАННЯ (ІСТОРІЯ)"])
    
    # --------------------------------------------------------------------------
    # ВКЛАДКА 1: ЧИСТИЙ ПРОГНОЗ НА МАЙБУТНЄ
    # --------------------------------------------------------------------------
    with view_tab_pred:
        if st.session_state.df_forecast_results is not None:
            st.subheader("🔮 Симуляція опадів на заданий період")
            st.write(f"Прогноз сформовано за допомогою моделі-переможця на період з **{st.session_state.forecast_start_date}** по **{st.session_state.forecast_end_date}**:")
            
            # Інтерактивна лінія тренду прогнозу
            fig_forecast_trend = px.line(
                st.session_state.df_forecast_results, 
                x="Дата", y="Ймовірність опадів (%)", markers=True,
                text="Ймовірність опадів (%)",
                title="Динаміка зміни ймовірності появи опадів за вердиктом моделі",
                color_discrete_sequence=["#e377c2"]
            )
            fig_forecast_trend.update_layout(yaxis_range=[-5, 105])
            fig_forecast_trend.update_traces(textposition="top center")
            st.plotly_chart(fig_forecast_trend, use_container_width=True)
            
            # Таблиця з результатами прогнозу
            st.dataframe(st.session_state.df_forecast_results, use_container_width=True)
        else:
            st.info("🔮 Тут з'явиться прогноз, щойно ти налаштуєш дати прогнозу та натиснеш кнопку запуску.")

    # --------------------------------------------------------------------------
    # ВКЛАДКА 2: ВСЕ ЩО СТОСУЄТЬСЯ ІСТОРІЇ, МОДЕЛЕЙ ТА ВИКАЧКИ ДАНИХ
    # --------------------------------------------------------------------------
    with view_tab_hist:
        if st.session_state.rf_metrics is not None:
            st.subheader("⚔️ Аналітичний батл алгоритмів (Hold-out Test)")
            
            metrics_df = pd.DataFrame({
                "Модель": ["Random Forest", "Random Forest", "Logistic Regression", "Logistic Regression", "Gradient Boosting", "Gradient Boosting"],
                "Метрика": ["F1-Score", "ROC AUC", "F1-Score", "ROC AUC", "F1-Score", "ROC AUC"],
                "Значення": [st.session_state.rf_metrics["F1"], st.session_state.rf_metrics["AUC"], st.session_state.lr_metrics["F1"], st.session_state.lr_metrics["AUC"], st.session_state.gb_metrics["F1"], st.session_state.gb_metrics["AUC"]]
            })
            
            fig_models = px.bar(metrics_df, x="Модель", y="Значення", color="Метрика", barmode="group", text_auto=".3f", color_discrete_sequence=["#1f77b4", "#ff7f0e"])
            st.plotly_chart(fig_models, use_container_width=True)
            
            # Оголошення переможця та параметри
            p_col1, p_col2 = st.columns([2, 1])
            with p_col1:
                st.info(f"🏆 Найкраща модель для поточної геолокації: **{st.session_state.best_model_name}**")
            with p_col2:
                with st.expander("⚙️ Переглянути підібрані гіперпараметри"):
                    st.json(st.session_state.best_params)
            
            # Графіки ознак
            g_col1, g_col2 = st.columns(2)
            with g_col1:
                st.plotly_chart(px.bar(st.session_state.feature_df, x="Важливість RF (%)", y="Ознака (Метеопараметр)", orientation='h', title="Важливість ознак (Random Forest)", color="Важливість RF (%)", color_continuous_scale="Viridis"), use_container_width=True)
            with g_col2:
                st.plotly_chart(px.bar(st.session_state.feature_df, x="Лінійна кореляція з таргетом (r)", y="Ознака (Метеопараметр)", orientation='h', title="Лінійна кореляція Пірсона (r) з таргатом", color="Лінійна кореляція з таргатом (r)", color_continuous_scale="RdBu", range_color=[-1,1]), use_container_width=True)
            
            st.markdown("---")
            st.subheader("📋 Робоча історична база даних (Вхідний датасет)")
            
            # ВІДНОВЛЕНА КНОПКА ВИКАЧКИ ДАНИХ ТА ФРАГМЕНТ ТАБЛИЦІ
            d_col1, d_col2 = st.columns([3, 1])
            with d_col1:
                st.write("Перші рядки історичної таблиці, завантаженої з архіву Open-Meteo:")
                st.dataframe(st.session_state.df_weather.head(5), use_container_width=True)
            with d_col2:
                st.write("📥 **Експорт даних у CSV:**")
                csv_buffer = st.session_state.df_weather.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Скачати поточний CSV файл",
                    data=csv_buffer,
                    file_name=f"historical_weather_data_{lat:.2f}_{lon:.2f}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
