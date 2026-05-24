import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight

def prepare_data(df: pd.DataFrame, threshold: float):
    """
    Оптимізована підготовка даних на основі ОФІЦІЙНОГО daily-списку Open-Meteo.
    Усуває витік даних та колінеарність.
    """
    df_clean = df.copy()
    
    # Динамічно формуємо бінарний таргет, якщо сума опадів є в датасеті (для історії)
    if 'precipitation_sum' in df_clean.columns:
        df_clean['target'] = (df_clean['precipitation_sum'] >= threshold).astype(int)
    
    # Feature Engineering: переводимо час у порядковий день року
    df_clean['time'] = pd.to_datetime(df_clean['time'])
    df_clean['dayofyear'] = df_clean['time'].dt.dayofyear
    
    # ОФІЦІЙНИЙ ВЕРИФІКОВАНИЙ НАБІР ОЗНАК (Синхронізовано зі скрапером на 100%)
    features_to_keep = [
        'shortwave_radiation_sum',
        'sunshine_duration',
        'daylight_duration',
        'apparent_temperature_max',
        'temperature_2m_mean',
        'wind_speed_10m_max',
        'wind_gusts_10m_max',
        'wind_direction_10m_dominant',
        'et0_fao_evapotranspiration',
        'dayofyear'
    ]
    
    X = df_clean[features_to_keep]
    
    # Безпечне розділення: якщо готуємо прогноз на майбутнє — таргету просто немає
    y = df_clean['target'] if 'target' in df_clean.columns else None
        
    return X, y

def train_and_compare_models(X, y):
    """
    Хронологічно розбиває дані, балансує ваги класів, оптимізує гіперпараметри
    для 3-х моделей на валідаційній вибірці та оцінює їх на Hold-out тесті.
    """
    # --- 1. ХРОНОЛОГІЧНЕ РОЗБИТТЯ НА 3 ПОСЛІДОВНІ БЛОКИ (60% / 20% / 20%) ---
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, shuffle=False)
    
    # --- 2. АВТОМАТИЧНЕ БАЛАНСУВАННЯ ВАГ КЛАСІВ ---
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    class_weights_dict = dict(zip(classes, weights))
    sample_weights_train = np.array([class_weights_dict[cls] for cls in y_train])
    
    # --- 3. МАСШТАБУВАННЯ ОЗНАК (Z-score нормалізація) ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # ==========================================================================
    # МОДЕЛЬ 1: LOGISTIC REGRESSION (Тюнінг сили регуляризації C)
    # ==========================================================================
    best_lr_c = 1.0
    best_lr_val_f1 = -1
    for c in [0.01, 0.1, 1.0, 10.0]:
        model = LogisticRegression(C=c, random_state=42, class_weight='balanced', max_iter=1000)
        model.fit(X_train_scaled, y_train)
        val_preds = model.predict(X_val_scaled)
        score = f1_score(y_val, val_preds, zero_division=0)
        if score > best_lr_val_f1:
            best_lr_val_f1 = score
            best_lr_c = c
            
    final_lr = LogisticRegression(C=best_lr_c, random_state=42, class_weight='balanced', max_iter=1000)
    final_lr.fit(np.vstack([X_train_scaled, X_val_scaled]), np.concatenate([y_train, y_val]))
    lr_preds = final_lr.predict(X_test_scaled)
    lr_metrics = {
        "F1": f1_score(y_test, lr_preds, zero_division=0),
        "P": precision_score(y_test, lr_preds, zero_division=0),
        "R": recall_score(y_test, lr_preds, zero_division=0),
        "AUC": roc_auc_score(y_test, final_lr.predict_proba(X_test_scaled)[:, 1])
    }

    # ==========================================================================
    # МОДЕЛЬ 2: RANDOM FOREST (Тюнінг максимальної глибини дерев)
    # ==========================================================================
    best_rf_depth = None
    best_rf_val_f1 = -1
    for depth in [5, 10, 15, None]:
        model = RandomForestClassifier(n_estimators=50, max_depth=depth, random_state=42, class_weight='balanced', n_jobs=-1)
        model.fit(X_train, y_train)
        val_preds = model.predict(X_val)
        score = f1_score(y_val, val_preds, zero_division=0)
        if score > best_rf_val_f1:
            best_rf_val_f1 = score
            best_rf_depth = depth
            
    final_rf = RandomForestClassifier(n_estimators=100, max_depth=best_rf_depth, random_state=42, class_weight='balanced', n_jobs=-1)
    final_rf.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))
    rf_preds = final_rf.predict(X_test)
    rf_metrics = {
        "F1": f1_score(y_test, rf_preds, zero_division=0),
        "P": precision_score(y_test, rf_preds, zero_division=0),
        "R": recall_score(y_test, rf_preds, zero_division=0),
        "AUC": roc_auc_score(y_test, final_rf.predict_proba(X_test)[:, 1])
    }

    # ==========================================================================
    # МОДЕЛЬ 3: GRADIENT BOOSTING (Тюнінг глибини бустингу)
    # ==========================================================================
    best_gb_depth = 3
    best_gb_val_f1 = -1
    for depth in [3, 5, 7]:
        model = HistGradientBoostingClassifier(max_depth=depth, random_state=42, max_iter=50)
        model.fit(X_train, y_train, sample_weight=sample_weights_train)
        val_preds = model.predict(X_val)
        score = f1_score(y_val, val_preds, zero_division=0)
        if score > best_gb_val_f1:
            best_gb_val_f1 = score
            best_gb_depth = depth
            
    final_gb = HistGradientBoostingClassifier(max_depth=best_gb_depth, random_state=42, max_iter=100)
    full_y_train_val = pd.concat([y_train, y_val])
    full_weights = compute_class_weight(class_weight='balanced', classes=classes, y=full_y_train_val)
    full_weights_dict = dict(zip(classes, full_weights))
    sample_weights_full = np.array([full_weights_dict[cls] for cls in full_y_train_val])
    
    final_gb.fit(pd.concat([X_train, X_val]), full_y_train_val, sample_weight=sample_weights_full)
    gb_preds = final_gb.predict(X_test)
    gb_metrics = {
        "F1": f1_score(y_test, gb_preds, zero_division=0),
        "P": precision_score(y_test, gb_preds, zero_division=0),
        "R": recall_score(y_test, gb_preds, zero_division=0),
        "AUC": roc_auc_score(y_test, final_gb.predict_proba(X_test)[:, 1])
    }

    # ==========================================================================
    # --- КОМПЛЕКСНА ОЦІНКА ОЗНАК (ДВА МЕТОДИ) ---
    # ==========================================================================
    importances = final_rf.feature_importances_
    full_df = X.copy()
    full_df['target'] = y
    correlations = full_df.corr()['target'].drop('target', errors='ignore')
    
    feature_analysis_df = pd.DataFrame({
        "Ознака (Метеопараметр)": X.columns,
        "Важливість RF (%)": np.round(importances * 100, 2),
        "Лінійна кореляція з таргатом (r)": np.round(correlations.values, 3)
    }).sort_values(by="Важливість RF (%)", ascending=False)
    
    # --- ВИЗНАЧЕННЯ НАЙКРАЩОЇ МОДЕЛІ ТА ЇЇ ГІПЕРПАРАМЕТРІВ ---
    scores = {
        "Random Forest": rf_metrics["F1"], 
        "Logistic Regression": lr_metrics["F1"], 
        "Gradient Boosting": gb_metrics["F1"]
    }
    best_model_name = max(scores, key=scores.get)
    
    # СИСТЕМНІ КЛЮЧІ (Англійською мовою, щоб app.py міг їх зчитати без збоїв)
    best_params = {}
    if best_model_name == "Random Forest":
        best_params = {"max_depth": best_rf_depth, "n_estimators": 100}
    elif best_model_name == "Logistic Regression":
        best_params = {"C": best_lr_c, "penalty": "L2"}
    elif best_model_name == "Gradient Boosting":
        best_params = {"max_depth": best_gb_depth, "max_iter": 100}
        
    return best_model_name, feature_analysis_df, rf_metrics, lr_metrics, gb_metrics, best_params

# ==============================================================================
# 🔮 ПРОДУКТОВИЙ ІНФЕРЕНС (ГЕНЕРАЦІЯ ПРОГНОЗУ НА МАЙБУТНЄ)
# ==============================================================================
def make_prediction(best_model, X_future):
    """
    Приймає навчену модель-переможця та очищену матрицю ознак для майбутнього періоду.
    Повертає вектор бінарних прогнозів (0 або 1) та масив відсоткових ймовірностей дощу.
    """
    predictions = best_model.predict(X_future)
    
    if hasattr(best_model, "predict_proba"):
        probabilities = best_model.predict_proba(X_future)[:, 1]
    else:
        probabilities = predictions.astype(float)
        
    return predictions, np.round(probabilities * 100, 1)