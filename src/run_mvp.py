# %% [markdown]
# ## Импорт библиотек

# %%
import polars as pl
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import rankdata

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer, StandardScaler, OneHotEncoder
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    f1_score, precision_score, recall_score, accuracy_score, balanced_accuracy_score
)

# %% [markdown]
# ## ШАГ 1. Загрузка исходных таблиц

# %%
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
data_path = f"{BASE_DIR}/data/raw/"

df_clients  = pl.read_csv(f"{data_path}clients.csv", try_parse_dates=True)
df_monthly  = pl.read_csv(f"{data_path}monthly_client.csv", try_parse_dates=True)
df_nso      = pl.read_csv(f"{data_path}monthly_nso.csv", try_parse_dates=True)
df_kp       = pl.read_csv(f"{data_path}kp.csv")
df_txns     = pl.read_csv(f"{data_path}transactions.csv", try_parse_dates=True)

df = (
    df_monthly
    .sort(["client_id", "report_date"])
    .with_columns(
        pl.col("report_date").rank("dense").cast(pl.Int32).alias("month_idx")
    )
)
max_idx = df["month_idx"].max()

# %% [markdown]
# ## ШАГ 2. Сегментированное бизнес-правило таргета ("Кто больше выводит")

# %%
df = df.with_columns(
    pl.col("total_liabilities_end").rolling_mean(3, min_samples=2)
      .cum_max().shift(1).over("client_id").alias("peak_liab"),
    pl.col("txn_cnt").cast(pl.Float64).rolling_mean(3, min_samples=2)
      .cum_max().shift(1).over("client_id").alias("peak_txn"),
    (pl.col("total_liabilities_end") - pl.col("total_liabilities_end").shift(1).over("client_id"))
      .alias("net_liab_change_1m")
)

# Дифференцированный отток по сегментам
df = df.with_columns(
    pl.when(pl.col("segment_name") == "A-Club").then(
        (pl.col("total_liabilities_end") < 0.75 * pl.col("peak_liab")) |
        (pl.col("net_liab_change_1m") < -5_000_000)
    )
    .when(pl.col("segment_name") == "Alfa Only").then(
        (pl.col("total_liabilities_end") < 0.65 * pl.col("peak_liab")) |
        (pl.col("net_liab_change_1m") < -1_500_000)
    )
    .otherwise(
        (pl.col("total_liabilities_end") < 0.50 * pl.col("peak_liab")) |
        (pl.col("txn_cnt") < 0.50 * pl.col("peak_txn"))
    )
    .alias("is_low")
).with_columns(
    (pl.col("is_low") & pl.col("is_low").shift(-1).over("client_id")).alias("is_low_confirmed")
).with_columns(
    pl.when(pl.col("is_low_confirmed")).then(pl.col("month_idx"))
      .min().over("client_id").alias("churn_idx_rule"),
    pl.col("month_idx").max().over("client_id").alias("last_idx"),
).with_columns(
    pl.min_horizontal(
        "churn_idx_rule",
        pl.when(pl.col("last_idx") < max_idx).then(pl.col("last_idx") + 1),
    ).alias("churn_idx")
).with_columns(
    pl.when(pl.col("month_idx") >= max_idx - 1).then(None)
      .when(pl.col("month_idx") + 1 == pl.col("churn_idx")).then(1)
      .otherwise(0)
      .cast(pl.Int8).alias("churn_next_month")
)

df = df.filter(
    pl.col("churn_idx").is_null() | (pl.col("month_idx") < pl.col("churn_idx"))
)

# %% [markdown]
# ## ШАГ 3. Объединение таблиц и числовое кодирование категорий

# %%
nso_agg = df_nso.group_by(["client_id", "report_date"]).agg(
    pl.col("early_close_flag").max().alias("nso_early_close_any"),
    pl.col("outflow_amt").sum().alias("nso_outflow_sum"),
)

df = (
    df.join(nso_agg, on=["client_id", "report_date"], how="left")
      .join(df_clients.select(["client_id", "registration_date", "employee_cnt_start"]),
            on="client_id", how="left")
      .sort(["client_id", "report_date"])
      .with_columns(
          pl.col("nso_early_close_any", "nso_outflow_sum").fill_null(0),
          ((pl.col("report_date") - pl.col("registration_date")).dt.total_days() / 30.4)
              .alias("client_age_m"),
          # Создаем гарантированно целочисленные признаки (Int32) для ML
          pl.col("segment_name").replace_strict(
              {"MASS": 0, "Alfa Only": 1, "A-Club": 2}, default=-1
          ).cast(pl.Int32).alias("segment_code"),
          pl.col("kp_id").str.slice(2).cast(pl.Int32, strict=False).fill_null(0).alias("kp_num")
      )
)

base_cols = ["total_liabilities_end", "demand_balance_avg", "nso_balance_avg", "txn_cnt",
             "txn_amt_in", "txn_amt_out", "active_days", "uniq_ca_cnt",
             "target_outflow_amt", "txn_oth_bank_amt", "salary_fund_amt"]

feat_exprs = []
for c in base_cols:
    x = pl.col(c).cast(pl.Float64)
    feat_exprs += [
        (x / (x.shift(1).over("client_id") + 1.0)).alias(f"{c}_chg_1m"),
        (x / (x.shift(3).over("client_id") + 1.0)).alias(f"{c}_chg_3m"),
        (x / (x.rolling_mean(6, min_samples=3).shift(1).over("client_id") + 1.0)).alias(f"{c}_vs_6m"),
        x.rolling_std(3, min_samples=2).over("client_id").alias(f"{c}_std_3m"),
    ]

feat_exprs += [
    (pl.col("total_liabilities_end") / (pl.col("peak_liab") + 1.0)).alias("liab_vs_peak"),
    (pl.col("txn_cnt") / (pl.col("peak_txn") + 1.0)).alias("txn_vs_peak"),
    (pl.col("total_liabilities_end") < pl.col("total_liabilities_end").shift(1))
        .cast(pl.Int8).rolling_sum(3, min_samples=1).over("client_id").alias("liab_down_months_3m"),
    ((pl.col("txn_amt_in") + 1.0) / (pl.col("txn_amt_out") + 1.0)).alias("cash_flow_ratio"),
    ((pl.col("salary_fund_amt").fill_null(0.0).clip(lower_bound=0.0) + 1.0) /
     (pl.col("total_liabilities_end").fill_null(0.0).clip(lower_bound=0.0) + 1.0)).alias("salary_share_of_liab"),
    (pl.col("total_liabilities_end") <= 100.0).cast(pl.Int8).alias("is_balance_zeroed")
]

df = df.with_columns(feat_exprs).with_columns(pl.col(pl.Boolean).cast(pl.Int8))


# %% [markdown]
# ## ШАГ 4. Хронологический сплит/разделение выборки (Все признаки строго числовые)

# %%
data = df.filter(pl.col("churn_next_month").is_not_null()).drop_nulls("peak_liab")

train = data.filter(pl.col("month_idx") <= max_idx - 13)
valid = data.filter(pl.col("month_idx").is_between(max_idx - 10, max_idx - 8))
test  = data.filter(pl.col("month_idx").is_between(max_idx - 5,  max_idx - 2))

# Исходные текстовые колонки kp_id и segment_name исключаются из матрицы X
drop_cols = [
    "client_id", "report_date", "cpr_id", "month_idx", "registration_date",
    "churn_idx", "churn_idx_rule", "last_idx", "is_low_confirmed", 
    "is_low", "peak_liab", "peak_txn", "churn_next_month", "target_outflow_amt", 
    "net_liab_change_1m", "segment_name", "kp_id"
]
features = [c for c in data.columns if c not in drop_cols and not c.startswith("target_outflow")]

X_tr = train.select(features).to_pandas()
y_tr = train["churn_next_month"].to_numpy()

X_va = valid.select(features).to_pandas()
y_va = valid["churn_next_month"].to_numpy()

X_te = test.select(features).to_pandas()
y_te = test["churn_next_month"].to_numpy()

# В признаках используются целочисленные коды категорий:
cat_features = ["segment_code", "kp_num"]


# %% [markdown]
# ## ШАГ 5. Препроцессинг для Sklearn моделей

# %%
num_cols = [c for c in features if c not in cat_features]

def _inf_to_nan(a):
    a = np.asarray(a, dtype="float64").copy()
    a[~np.isfinite(a)] = np.nan
    return a

def _signed_log(a):
    return np.sign(a) * np.log1p(np.abs(a))

prep_tree = ColumnTransformer([
    ("num", Pipeline([("inf", FunctionTransformer(_inf_to_nan, feature_names_out="one-to-one")),
                      ("imp", SimpleImputer(strategy="median"))]), num_cols),
    ("cat", SimpleImputer(strategy="most_frequent"), cat_features)
], remainder="drop")

Xt_tr = prep_tree.fit_transform(X_tr)
Xt_va = prep_tree.transform(X_va)
Xt_te = prep_tree.transform(X_te)

prep_lin = ColumnTransformer([
    ("num", Pipeline([("inf", FunctionTransformer(_inf_to_nan, feature_names_out="one-to-one")),
                      ("imp", SimpleImputer(strategy="median")),
                      ("log", FunctionTransformer(_signed_log, feature_names_out="one-to-one")),
                      ("sc",  StandardScaler())]), num_cols),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat_features)
], remainder="drop")

Xl_tr = prep_lin.fit_transform(X_tr)
Xl_va = prep_lin.transform(X_va)
Xl_te = prep_lin.transform(X_te)

# %% [markdown]
# ## ШАГ 6. Обучение моделей (LightGBM с весами, RF, LogReg)

# %%
# Расчет весов без NaN и отрицательных значений
raw_liab = train["total_liabilities_end"].fill_null(0.0).to_numpy()
sample_weights_tr = np.log1p(np.maximum(0.0, raw_liab))
sample_weights_tr = np.nan_to_num(np.clip(sample_weights_tr, 1.0, None), nan=1.0)

# 1. LightGBM с финансовым штрафом
model_lgb = lgb.LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.03,
    num_leaves=31,
    min_child_samples=30,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbose=-1
)
model_lgb.fit(
    X_tr, y_tr,
    sample_weight=sample_weights_tr,
    eval_set=[(X_va, y_va)],
    eval_metric="auc",
    categorical_feature=cat_features,
    callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)]
)

# 2. Random Forest
rf = RandomForestClassifier(
    n_estimators=300, min_samples_leaf=5, max_features="sqrt",
    class_weight="balanced_subsample", n_jobs=-1, random_state=42
)
rf.fit(Xt_tr, y_tr)

# 3. Logistic Regression
lr = LogisticRegression(C=0.1, class_weight="balanced", max_iter=3000, random_state=42)
lr.fit(Xl_tr, y_tr)


# %% [markdown]
# ## ШАГ 7. Сборка ансамбля и сравнение метрик

# %%
p_lgb_va, p_lgb_te = model_lgb.predict_proba(X_va)[:, 1], model_lgb.predict_proba(X_te)[:, 1]
p_rf_va,  p_rf_te  = rf.predict_proba(Xt_va)[:, 1],        rf.predict_proba(Xt_te)[:, 1]
p_lr_va,  p_lr_te  = lr.predict_proba(Xl_va)[:, 1],        lr.predict_proba(Xl_te)[:, 1]

# Ранговый блендинг (0.5 LGB + 0.3 RF + 0.2 LR)
rank_va = (
    0.5 * (rankdata(p_lgb_va) / len(p_lgb_va)) +
    0.3 * (rankdata(p_rf_va)  / len(p_rf_va))  +
    0.2 * (rankdata(p_lr_va)  / len(p_lr_va))
)
rank_te = (
    0.5 * (rankdata(p_lgb_te) / len(p_lgb_te)) +
    0.3 * (rankdata(p_rf_te)  / len(p_rf_te))  +
    0.2 * (rankdata(p_lr_te)  / len(p_lr_te))
)

results = {}
rng = np.random.default_rng(42)

def evaluate_model(name, p_va, p_te):
    prec, rec, thr = precision_recall_curve(y_va, p_va)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    t = thr[np.argmax(f1[:-1])]
    pred = (p_te >= t).astype(int)

    k = int(0.1 * len(p_te))
    top = np.argsort(-(p_te + rng.uniform(0, 1e-9, len(p_te))))[:k]

    results[name] = {
        "Accuracy":          round(100 * accuracy_score(y_te, pred), 2),
        "Balanced accuracy": round(100 * balanced_accuracy_score(y_te, pred), 2),
        "Precision":         round(100 * precision_score(y_te, pred, zero_division=0), 2),
        "Recall":            round(100 * recall_score(y_te, pred), 2),
        "F1":                round(100 * f1_score(y_te, pred), 2),
        "ROC-AUC":           round(100 * roc_auc_score(y_te, p_te), 2),
        "PR-AUC":            round(100 * average_precision_score(y_te, p_te), 2),
        "Recall@10%":        round(100 * (y_te[top].sum() / y_te.sum()), 2),
        "Precision@10%":     round(100 * y_te[top].mean(), 2),
        "Порог":             round(float(t), 4)
    }

evaluate_model("LightGBM (Cost-Sensitive)", p_lgb_va, p_lgb_te)
evaluate_model("Random Forest", p_rf_va, p_rf_te)
evaluate_model("Logistic Regression", p_lr_va, p_lr_te)
evaluate_model("Ensemble (0.5 LGB + 0.3 RF + 0.2 LR)", rank_va, rank_te)

print("\n--- ИТОГОВЫЕ МЕТРИКИ НА ТЕСТЕ (%) ---")
print(pd.DataFrame(results).T.to_string())

# %% [markdown]
# ## ШАГ 8. Бизнес-правила и формирование CRM-витрины

# %%
current_data = df.filter(pl.col("month_idx") == max_idx)
current_report_date = current_data.select(pl.col("report_date").max()).item()

txns_recency = (
    df_txns
    .filter(pl.col("txn_date") <= current_report_date)
    .group_by("client_id")
    .agg(
        (pl.lit(current_report_date) - pl.col("txn_date").max()).dt.total_days().alias("days_since_last_txn")
    )
)
X_curr = current_data.select(features).to_pandas()

# 1. Берем честную вероятность от лучшей модели (LightGBM), а не ранги
p_curr_lgb = model_lgb.predict_proba(X_curr)[:, 1]

# 2. Суммируем SHAP-вклады по смысловым группам (исключая статичные сегмент и отрасль)
contribs = model_lgb.predict(X_curr, pred_contrib=True)[:, :-1]

DOMAIN_PREFIXES = {
    "oth_bank":       ("txn_oth_bank",),
    "nso":            ("nso_",),
    "salary":         ("salary_",),
    "txn_activity":   ("txn_cnt", "txn_vs_peak", "active_days"),
    "cash_flow":      ("txn_amt", "cash_flow_ratio"),
    "counterparties": ("uniq_ca_cnt",),
    "balances":       ("total_liabilities", "demand_balance", "liab_", "is_balance_zeroed"),
}

domain_names = list(DOMAIN_PREFIXES.keys())
domain_scores = np.column_stack([
    contribs[:, [i for i, f in enumerate(features) if f.startswith(prefs)]].sum(axis=1)
    for prefs in DOMAIN_PREFIXES.values()
])

best_idx = np.argmax(domain_scores, axis=1)
best_val = np.max(domain_scores, axis=1)
top_feat_names = [domain_names[i] if v > 0 else "general" for i, v in zip(best_idx, best_val)]

REASONS_MAP = {
    "oth_bank":       ("Рост переводов в сторонние банки", "Кэшбэк за обороты и спецтариф на межбанковские платежи"),
    "nso":            ("Вывод средств или досрочное закрытие НСО", "Индивидуальная повышенная ставка на депозит / НСО"),
    "salary":         ("Сокращение зарплатного фонда", "Бонусные условия и бесплатное обслуживание зарплатного проекта"),
    "txn_activity":   ("Падение транзакционной активности", "Аудит расчетного обслуживания и скидка на пакет РКО"),
    "cash_flow":      ("Списания превышают поступления (отток оборота)", "Предложение овердрафта или кредитной линии на пополнение оборотных средств"),
    "counterparties": ("Сокращение числа контрагентов по счету", "Бесплатная проверка контрагентов и льготы на ВЭД / валютный контроль"),
    "balances":       ("Систематическое снижение остатков на счетах", "Процент на неснижаемый остаток по расчетному счету"),
    "general":        ("Комплексное отклонение поведенческих метрик", "Индивидуальный пересмотр тарифа РКО и пакетных опций"),
}

def map_reason_and_action(domain: str, segment: str) -> tuple[str, str]:
    reason, offer = REASONS_MAP[domain]
    if segment == "A-Club":
        channel = "Личная встреча / звонок VIP-менеджера: "
    elif segment == "Alfa Only":
        channel = "Звонок персонального менеджера: "
    else:
        channel = "Таргетированное предложение в Банк-Клиенте: "
    return reason, channel + offer

mapped = [
    map_reason_and_action(d, s)
    for d, s in zip(top_feat_names, current_data["segment_name"].to_list())
]
reasons = [m[0] for m in mapped]
actions = [m[1] for m in mapped]

# 3. Собираем таблицу и применяем бизнес-правила фильтрации
crm_export_table = (
    current_data
    .with_columns([
        pl.Series("churn_probability", p_curr_lgb),
        pl.Series("top_risk_feature", top_feat_names),
        pl.Series("churn_reason", reasons),
        pl.Series("recommended_action", actions),
        pl.col("report_date").cast(pl.Date).cast(pl.Utf8).alias("scored_date"),
    ])
    .join(df_kp.select(["kp_id", "kp_name"]), on="kp_id", how="left")
    .join(txns_recency, on="client_id", how="left")
    .with_columns([
        pl.col("days_since_last_txn").fill_null(999),
        (pl.col("churn_probability") * pl.col("total_liabilities_end")).alias("expected_loss_rub"),
    ])
    # ПРАВИЛО 1 (Исключения): убираем уже пустые счета и новичков младше 3 месяцев
    .filter(
        (pl.col("is_balance_zeroed") == 0) &
        (pl.col("client_age_m") >= 3.0) &
        (pl.col("total_liabilities_end") > 10_000)
    )
    # ПРАВИЛО 2 (Порог риска): отсекаем вероятность < 6% перед ранжированием по expected_loss_rub
    .filter(pl.col("churn_probability") >= 0.06)
    # ПРАВИЛО 3 (Матрица приоритетов): делим по сумме под риском и сегменту
    .with_columns(
        pl.when((pl.col("expected_loss_rub") >= 3_000_000) | ((pl.col("segment_name") == "A-Club") & (pl.col("churn_probability") >= 0.25)))
          .then(pl.lit("1. КРИТИЧЕСКИЙ"))
          .when((pl.col("expected_loss_rub") >= 500_000) | (pl.col("segment_name") == "Alfa Only"))
          .then(pl.lit("2. ВЫСОКИЙ"))
          .otherwise(pl.lit("3. СТАНДАРТНЫЙ"))
          .alias("retention_group")
    )
    # ПРАВИЛО 4 (Лимит пропускной способности): отдаем в CRM топ-50 самых дорогих по рискам клиентов
    .sort("expected_loss_rub", descending=True)
    .head(50)
    .select([
        "scored_date",
        "client_id",
        "segment_name",
        "kp_name",
        pl.col("total_liabilities_end").round(0).alias("current_liabilities_rub"),
        pl.col("churn_probability").round(3),
        pl.col("expected_loss_rub").round(0),
        "days_since_last_txn",
        "retention_group",
        "churn_reason",
        "recommended_action",
    ])
)

print(f"\n--- CRM-ВИТРИНА СФОРМИРОВАНА (Взято в работу: {crm_export_table.height} клиентов) ---")
print(crm_export_table.head(10))

# %% [markdown]
# ## ШАГ 9. Выгрузка данных в БД для создания дашборда

# %%
import os
from datetime import datetime
from dotenv import load_dotenv, find_dotenv
from sqlalchemy import create_engine

load_dotenv(find_dotenv())

crm_pretty = crm_export_table.select([
    pl.col("scored_date").alias("Дата скоринга"),
    pl.col("client_id").alias("ID клиента"),
    pl.col("segment_name").alias("Сегмент"),
    pl.col("kp_name").alias("Отрасль"),
    pl.col("current_liabilities_rub").alias("Текущие пассивы (руб)"),
    (pl.col("churn_probability") * 100).round(1).alias("Риск оттока (%)"),
    pl.col("expected_loss_rub").alias("Ожидаемые потери (руб)"),
    pl.col("days_since_last_txn").alias("Дней без транзакций"),
    pl.col("retention_group").alias("Приоритет"),
    pl.col("churn_reason").alias("Причина риска"),
    pl.col("recommended_action").alias("Рекомендуемое действие")
])

engine = create_engine(
    f"postgresql+psycopg2://{os.environ['POSTGRES_USER']}:{os.environ['POSTGRES_PASSWORD']}@localhost:5432/{os.environ['POSTGRES_DB']}"
)

df_out = crm_pretty.to_pandas()
df_out.to_sql("crm_mvp_mart", engine, if_exists="replace", index=False)

df_hist = df_out.copy()
df_hist["Время запуска"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
df_hist.to_sql("crm_mvp_history", engine, if_exists="append", index=False)

print(f"ГОТОВО! Витрина crm_mvp_mart ({len(df_out)} строк) обновлена, история сохранена!")


