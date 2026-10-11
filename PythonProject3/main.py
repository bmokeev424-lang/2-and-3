import os
import pandas as pd
from utils import generate_data, show_basic_stats, show_plots
from models import fuzzy_state, train_neural, predict_neural, STATE_NAMES
def evaluate_new_department(model, scaler):
    print("\nВведите данные нового подразделения:")
    revenue = float(input("Выручка: "))
    costs = float(input("Затраты: "))
    employees = int(input("Численность: "))
    productivity = float(input("Производительность: "))
    overdue_orders = int(input("Просроченные заказы: "))
    defects = int(input("Дефекты: "))
    plan_percent = float(input("Выполнение плана, %: "))
    profitability = (revenue - costs) / revenue * 100
    revenue_per_employee = revenue / employees
    print(f"\nРентабельность: {profitability:.2f}%")
    print(f"Выручка на сотрудника: {revenue_per_employee:.3f}")
    fuzzy_res, fuzzy_scores = fuzzy_state(
        profitability, plan_percent, overdue_orders, defects
    )
    row = {
        "revenue": revenue, "costs": costs, "employees": employees,
        "productivity": productivity, "overdue_orders": overdue_orders,
        "defects": defects, "plan_percent": plan_percent,
        "profitability": profitability,
        "revenue_per_employee": revenue_per_employee
    }
    neural_res, neural_proba = predict_neural(model, scaler, row)
    print("\nНечеткая модель:", STATE_NAMES[fuzzy_res], fuzzy_scores)
    print("Нейронная сеть:", STATE_NAMES[neural_res], neural_proba)
    if fuzzy_res == neural_res:
        print("Итог: результаты совпали.")
    else:
        print("Итог: результаты разошлись, нужен доп. анализ.")
def main():
    df = None
    model = None
    scaler = None
    metrics = None
    while True:
        print("\nМеню:")
        print("1. Сгенерировать / загрузить данные")
        print("2. Показать основные показатели")
        print("3. Показать графики")
        print("4. Проверить нечеткую модель")
        print("5. Обучить нейронную сеть")
        print("6. Показать метрики и матрицу ошибок")
        print("7. Оценить новое подразделение")
        print("8. Сравнить два метода")
        print("0. Выход")
        choice = input("Выберите пункт: ")
        if choice == "1":
            if os.path.exists("data.csv"):
                df = pd.read_csv("data.csv")
                print("Данные загружены из data.csv")
            else:
                df = generate_data()
                print("Данные сгенерированы и сохранены в data.csv")
        elif choice == "2":
            if df is None:
                print("Сначала загрузите данные.")
            else:
                show_basic_stats(df)
        elif choice == "3":
            if df is None:
                print("Сначала загрузите данные.")
            else:
                show_plots(df)
        elif choice == "4":
            if df is None:
                print("Сначала загрузите данные.")
            else:
                row = df.iloc[0]
                res, scores = fuzzy_state(
                    row["profitability"], row["plan_percent"],
                    row["overdue_orders"], row["defects"]
                )
                print("Категория:", STATE_NAMES[res], scores)
        elif choice == "5":
            if df is None:
                print("Сначала загрузите данные.")
            else:
                model, scaler, metrics = train_neural(df)
                print("Нейронная сеть обучена.")
        elif choice == "6":
            if model is None:
                print("Сначала обучите нейронную сеть.")
            else:
                print("Accuracy:", metrics["accuracy"])
                print("Precision:", metrics["precision"])
                print("Recall:", metrics["recall"])
                print("F1:", metrics["f1"])
                print("Матрица ошибок:\n", metrics["confusion"])
        elif choice == "7":
            if model is None:
                print("Сначала обучите нейронную сеть.")
            else:
                evaluate_new_department(model, scaler)
        elif choice == "8":
            if model is None:
                print("Сначала обучите нейронную сеть.")
            else:
                evaluate_new_department(model, scaler)
        elif choice == "0":
            print("Выход.")
            break
        else:
            print("Неверный пункт.")
if __name__ == "__main__":
    main()