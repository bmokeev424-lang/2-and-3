import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
FEATURES = [
    "revenue", "costs", "employees", "productivity",
    "overdue_orders", "defects", "plan_percent",
    "profitability", "revenue_per_employee"
]
STATE_NAMES = {0: "Проблемное", 1: "Требует внимания", 2: "Устойчивое"}

def trimf(x, a, b, c):
    if x <= a or x >= c:
        return 0.0
    if x == b:
        return 1.0
    if x < b:
        return (x - a) / (b - a)
    return (c - x) / (c - b)

def fuzzy_state(profitability, plan_percent, overdue_orders, defects):
    p_low = trimf(profitability, -50, 0, 15)
    p_mid = trimf(profitability, 5, 20, 35)
    p_high = trimf(profitability, 25, 40, 60)
    plan_low = trimf(plan_percent, 50, 70, 95)
    plan_mid = trimf(plan_percent, 85, 100, 110)
    plan_high = trimf(plan_percent, 100, 115, 130)
    over_low = trimf(overdue_orders, -5, 0, 10)
    over_mid = trimf(overdue_orders, 5, 15, 25)
    over_high = trimf(overdue_orders, 20, 30, 45)
    def_low = trimf(defects, -5, 0, 8)
    def_mid = trimf(defects, 5, 12, 20)
    def_high = trimf(defects, 15, 22, 30)
    rules = [
        (min(p_high, plan_high, over_low, def_low), 2),
        (min(p_high, plan_mid, over_low, def_low), 2),
        (min(p_mid, plan_high, over_low, def_low), 2),
        (min(p_mid, plan_mid, over_mid, def_mid), 1),
        (min(p_low, plan_high, over_mid, def_mid), 1),
        (min(p_mid, plan_low, over_low, def_low), 1),
        (min(p_high, plan_low, over_mid, def_mid), 1),
        (min(p_low, plan_low, over_high, def_high), 0),
        (min(p_low, plan_mid, over_high, def_mid), 0),
        (min(p_mid, plan_low, over_high, def_high), 0),
    ]
    scores = {0: 0.0, 1: 0.0, 2: 0.0}
    for strength, state in rules:
        if strength > scores[state]:
            scores[state] = strength
    best = max(scores, key=scores.get)
    return best, scores

def train_neural(df):
    X = df[FEATURES]
    y = df["state"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32 ),
        activation="relu",
        solver="adam",
        max_iter=1000,
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "confusion": confusion_matrix(y_test, y_pred),
    }
    return model, scaler, metrics
def predict_neural(model, scaler, row):
    X = pd.DataFrame([row])[FEATURES]
    X = scaler.transform(X)
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    return pred, proba