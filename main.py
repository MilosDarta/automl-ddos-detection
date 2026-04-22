#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Сравнение AutoML-фреймворков для обнаружения DDoS-атак на наборе данных CIC-DDoS2019.

Скрипт выполняет:
1. Загрузку и объединение parquet-файлов
2. Предобработку (удаление идентификаторов, низковариативных и коррелированных признаков,
   объединение редких классов)
3. Анализ распределения классов
4. Стратифицированное разделение на train/val/test
5. Обучение ручного бейзлайна LightGBM с весами классов
6. Обучение AutoGluon
7. Обучение LightAutoML
8. Сравнение результатов по macro F1-score и времени обучения

Для запуска необходимы библиотеки из requirements.txt.
Данные должны находиться в папке data/ в виде .parquet файлов.
"""

import os
import sys
import time
import glob
import warnings
import subprocess
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.utils.class_weight import compute_class_weight
import lightgbm as lgb

# Подавление предупреждений для чистоты вывода
warnings.filterwarnings('ignore')

# ============================================================
# Проверка и установка зависимостей (для Colab)
# ============================================================
def install_requirements():
    """Установка необходимых пакетов, если их нет."""
    required = {
        'autogluon': 'autogluon',
        'lightautoml': 'lightautoml',
        'lightgbm': 'lightgbm',
        'sklearn': 'scikit-learn',
        'pandas': 'pandas',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'pyarrow': 'pyarrow'
    }
    missing = []
    for module, package in required.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(package)
    if missing:
        print(f"Установка отсутствующих пакетов: {missing}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + missing)

install_requirements()

# Импорт AutoML после установки
from autogluon.tabular import TabularDataset, TabularPredictor
from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.tasks import Task

# ============================================================
# Конфигурация
# ============================================================
DATA_DIR = 'data/'
TARGET = 'Label'
RANDOM_STATE = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.2  # от оставшихся 80%
TIMEOUT = 1800   # 30 минут для AutoML

# ============================================================
# 1. Загрузка данных
# ============================================================
def load_data(data_dir: str) -> pd.DataFrame:
    """Загружает и объединяет все parquet-файлы из указанной директории."""
    parquet_files = glob.glob(os.path.join(data_dir, '*.parquet'))
    if not parquet_files:
        raise FileNotFoundError(f"Не найдено .parquet файлов в {data_dir}")
    print(f"Найдено {len(parquet_files)} parquet-файлов")
    df = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)
    print(f"Объединённый датасет: {df.shape[0]:,} строк, {df.shape[1]} столбцов")
    return df

# ============================================================
# 2. Предобработка
# ============================================================
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Очистка данных: удаление идентификаторов, низковариативных и коррелированных признаков,
       объединение редких классов."""
    # Удаление идентификаторов
    drop_cols = ['Flow ID', 'Src IP', 'Dst IP', 'Src Port', 'Dst Port', 'Timestamp']
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True, errors='ignore')

    # Низковариативные признаки
    low_var = [
        'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags', 'FIN Flag Count',
        'PSH Flag Count', 'ECE Flag Count', 'Fwd Avg Bytes/Bulk',
        'Fwd Avg Packets/Bulk', 'Fwd Avg Bulk Rate', 'Bwd Avg Bytes/Bulk',
        'Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate'
    ]
    df.drop(columns=[c for c in low_var if c in df.columns], inplace=True, errors='ignore')

    # Коррелированные признаки
    redundant = [
        'RST Flag Count', 'Fwd IAT Min', 'Fwd IAT Max', 'Subflow Bwd Bytes',
        'Subflow Fwd Bytes', 'Fwd IAT Total', 'Fwd Packets/s', 'Packet Length Min',
        'Avg Packet Size', 'Fwd IAT Std', 'Idle Max'
    ]
    df.drop(columns=[c for c in redundant if c in df.columns], inplace=True, errors='ignore')

    # Бесконечности и NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    # Объединение редких классов (менее 200 примеров)
    label_counts = df[TARGET].value_counts()
    rare_labels = label_counts[label_counts < 200].index.tolist()
    if rare_labels:
        print(f"Объединение редких классов: {rare_labels}")
        df[TARGET] = df[TARGET].apply(lambda x: 'Rare_Attack' if x in rare_labels else x)

    df[TARGET] = df[TARGET].astype('category')
    print(f"После предобработки: {df.shape[0]:,} строк, {df.shape[1]} столбцов")
    print("Классы:", df[TARGET].cat.categories.tolist())
    return df

# ============================================================
# 3. Анализ распределения классов (гистограмма)
# ============================================================
def plot_class_distribution(df: pd.DataFrame):
    """Строит горизонтальную столбчатую диаграмму распределения классов."""
    counts = df[TARGET].value_counts()
    plt.figure(figsize=(10, 6))
    sns.barplot(x=counts.values, y=counts.index, palette='viridis')
    plt.title('Распределение классов в CIC-DDoS2019')
    plt.xlabel('Количество примеров')
    plt.ylabel('Класс')
    plt.tight_layout()
    plt.savefig('class_distribution.png', dpi=150)
    plt.show()
    print("Гистограмма сохранена как 'class_distribution.png'")

# ============================================================
# 4. Разделение на train/val/test
# ============================================================
def split_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Стратифицированное разделение: train (64%), val (16%), test (20%)."""
    train_val, test = train_test_split(
        df, test_size=TEST_SIZE, stratify=df[TARGET], random_state=RANDOM_STATE
    )
    train, val = train_test_split(
        train_val, test_size=VAL_SIZE, stratify=train_val[TARGET], random_state=RANDOM_STATE
    )
    print(f"Train: {train.shape}, Val: {val.shape}, Test: {test.shape}")
    return train, val, test

# ============================================================
# 5. Оценка модели
# ============================================================
def evaluate_model(name: str, y_true: np.ndarray, y_pred: np.ndarray, train_time: float) -> Dict:
    """Вычисляет метрики и возвращает словарь с результатами."""
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    print(f"\n{'='*50}")
    print(f"{name}")
    print(f"{'='*50}")
    print(f"Время обучения: {train_time:.2f} сек")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision (macro): {prec:.4f}")
    print(f"Recall (macro): {rec:.4f}")
    print(f"F1-score (macro): {f1:.4f}")
    print("\nClassification Report (первые 5 классов):")
    # Ограничим вывод для краткости
    report = classification_report(y_true, y_pred, zero_division=0)
    lines = report.split('\n')
    print('\n'.join(lines[:15]))
    return {
        'Framework': name,
        'Accuracy': acc,
        'Precision (macro)': prec,
        'Recall (macro)': rec,
        'F1 (macro)': f1,
        'Time': train_time
    }

# ============================================================
# 6. LightGBM Baseline
# ============================================================
def train_lightgbm(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> Tuple[np.ndarray, float]:
    """Обучение LightGBM с весами классов."""
    le = LabelEncoder()
    y_train = le.fit_transform(train[TARGET])
    y_val = le.transform(val[TARGET])
    y_test = le.transform(test[TARGET])

    feature_cols = [c for c in train.columns if c != TARGET]
    X_train = train[feature_cols]
    X_val = val[feature_cols]
    X_test = test[feature_cols]

    # Веса классов
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    weight_dict = dict(zip(np.unique(y_train), class_weights))
    sample_weights = np.array([weight_dict[label] for label in y_train])

    params = {
        'objective': 'multiclass',
        'num_class': len(le.classes_),
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'seed': RANDOM_STATE
    }

    print("\nОбучение LightGBM Baseline...")
    start = time.time()
    model = lgb.train(
        params,
        lgb.Dataset(X_train, label=y_train, weight=sample_weights),
        valid_sets=[lgb.Dataset(X_val, label=y_val)],
        num_boost_round=200,
        callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
    )
    train_time = time.time() - start

    y_pred_proba = model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)
    # Инверсное преобразование для строковых меток
    y_pred_labels = le.inverse_transform(y_pred)
    return y_pred_labels, train_time

# ============================================================
# 7. AutoGluon
# ============================================================
def train_autogluon(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> Tuple[np.ndarray, float]:
    """Обучение AutoGluon с пресетом medium_quality."""
    train_ag = TabularDataset(train)
    val_ag = TabularDataset(val)
    test_ag = TabularDataset(test)

    predictor = TabularPredictor(
        label=TARGET,
        problem_type='multiclass',
        eval_metric='f1_macro'
    )

    print("\nОбучение AutoGluon...")
    start = time.time()
    predictor.fit(
        train_ag,
        tuning_data=val_ag,
        time_limit=TIMEOUT,
        presets='medium_quality',
        excluded_model_types=['RF', 'XT']  # избегаем проблем с памятью
    )
    train_time = time.time() - start

    y_pred = predictor.predict(test_ag)
    return y_pred.values, train_time

# ============================================================
# 8. LightAutoML
# ============================================================
def train_lightautoml(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> Tuple[np.ndarray, float]:
    """Обучение LightAutoML с исправлением маппинга меток."""
    task = Task(name='multiclass', metric='accuracy')
    automl = TabularAutoML(
        task=task,
        timeout=TIMEOUT,
        cpu_limit=4,
        general_params={'seed': RANDOM_STATE}
    )

    print("\nОбучение LightAutoML...")
    start = time.time()
    _ = automl.fit_predict(
        train,
        roles={'target': TARGET}
    )
    train_time = time.time() - start

    # Получение предсказаний и исправление маппинга через валидацию
    val_proba = automl.predict(val).data
    val_pred_idx = np.argmax(val_proba, axis=1)
    val_true = val[TARGET].values

    # Построение маппинга индекс -> метка
    index_to_label = {}
    for idx in np.unique(val_pred_idx):
        mask = val_pred_idx == idx
        labels, counts = np.unique(val_true[mask], return_counts=True)
        index_to_label[idx] = labels[np.argmax(counts)]

    n_classes = val_proba.shape[1]
    full_mapping = {i: index_to_label.get(i, 'Unknown') for i in range(n_classes)}

    # Применение к тесту
    test_proba = automl.predict(test).data
    test_pred_idx = np.argmax(test_proba, axis=1)
    y_pred = np.array([full_mapping[i] for i in test_pred_idx])

    return y_pred, train_time

# ============================================================
# 9. Визуализация сравнения
# ============================================================
def plot_comparison(results: List[Dict]):
    """Строит столбчатые диаграммы для F1-macro и времени обучения."""
    df = pd.DataFrame(results).round(4)
    print("\n" + "="*60)
    print("ИТОГОВОЕ СРАВНЕНИЕ")
    print("="*60)
    print(df.to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # F1-macro
    sns.barplot(data=df, x='Framework', y='F1 (macro)', ax=axes[0], palette='viridis')
    axes[0].set_title('F1-score (macro)')
    axes[0].set_ylim(0.7, 0.85)

    # Время
    sns.barplot(data=df, x='Framework', y='Time', ax=axes[1], palette='rocket')
    axes[1].set_title('Время обучения (сек)')

    plt.tight_layout()
    plt.savefig('comparison_results.png', dpi=150)
    plt.show()
    print("График сравнения сохранён как 'comparison_results.png'")

# ============================================================
# MAIN
# ============================================================
def main():
    print("="*60)
    print("Сравнение AutoML для обнаружения DDoS-атак (CIC-DDoS2019)")
    print("="*60)

    # 1. Загрузка данных
    df = load_data(DATA_DIR)

    # 2. Предобработка
    df = preprocess_data(df)

    # 3. Визуализация распределения
    plot_class_distribution(df)

    # 4. Разделение
    train, val, test = split_data(df)

    # 5. Список для результатов
    results = []

    # 6. LightGBM Baseline
    y_pred_lgb, time_lgb = train_lightgbm(train, val, test)
    results.append(evaluate_model("LightGBM Baseline", test[TARGET].values, y_pred_lgb, time_lgb))

    # 7. AutoGluon
    y_pred_ag, time_ag = train_autogluon(train, val, test)
    results.append(evaluate_model("AutoGluon", test[TARGET].values, y_pred_ag, time_ag))

    # 8. LightAutoML
    y_pred_lama, time_lama = train_lightautoml(train, val, test)
    results.append(evaluate_model("LightAutoML", test[TARGET].values, y_pred_lama, time_lama))

    # 9. Визуализация сравнения
    plot_comparison(results)

    print("\nГотово!")

if __name__ == "__main__":
    main()
