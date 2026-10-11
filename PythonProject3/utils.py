import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def generate_data(seed=42):
    np.random.seed(seed)

    departments = [f"Цех {i:02d}" for i in range(1, 21)]
    periods = pd.date_range("2022-01-01", periods=60, freq="MS")

    rows = []

    for dep in departments:
        for period in periods:
            revenue = np.random.uniform(1.0, 20.0)
            costs = np.random.uniform(0.6, 18.0)
            employees = np.random.randint(10, 251)
            productivity = np.random.uniform(40, 180)
            overdue_orders = np.random.randint(0, 41)
            defects = np.random.randint(0, 26)
            plan_percent = np.random.uniform(70, 125)

            profitability = (revenue - costs) / revenue * 100
            revenue_per_employee = revenue / employees

            state = make_state(profitability, plan_percent, overdue_orders, defects)

            rows.append({
                "period": period.strftime("%Y-%m"),
                "department": dep,
                "revenue": round(revenue, 3),
                "costs": round(costs, 3),
                "employees": employees,
                "productivity": round(productivity, 2),
                "overdue_orders": overdue_orders,
                "defects": defects,
                "plan_percent": round(plan_percent, 2),
                "profitability": round(profitability, 2),
                "revenue_per_employee": round(revenue_per_employee, 3),
                "state": state
            })

    df = pd.DataFrame(rows)
    df.to_csv("data.csv", index=False)
    return df


def make_state(profitability, plan_percent, overdue_orders, defects):
    score = 0

    if profitability >= 20:
        score += 2
    elif profitability >= 10:
        score += 1

    if plan_percent >= 100:
        score += 2
    elif plan_percent >= 90:
        score += 1

    if overdue_orders <= 5:
        score += 2
    elif overdue_orders <= 15:
        score += 1

    if defects <= 3:
        score += 2
    elif defects <= 10:
        score += 1

    if score >= 7:
        return 2
    elif score >= 4:
        return 1
    else:
        return 0


def show_basic_stats(df):
    print("\nОсновные показатели:")
    print(df.describe())
    print("\nРаспределение state:")
    print(df["state"].value_counts().sort_index())


def show_plots(df):
    plt.style.use("seaborn-v0_8-whitegrid")

    state_names = {0: "Проблемное", 1: "Требует внимания", 2: "Устойчивое"}
    colors = ["#d62728", "#ff7f0e", "#2ca02c"]

    plt.figure(figsize=(8, 5))
    means = df.groupby("state")["profitability"].mean()
    bars = plt.bar(
        [state_names[s] for s in means.index],
        means.values,
        color=colors
    )
    plt.title("Средняя рентабельность по состояниям подразделений")
    plt.xlabel("Состояние")
    plt.ylabel("Рентабельность, %")
    plt.grid(axis="y", alpha=0.4)
    for b, v in zip(bars, means.values):
        plt.text(b.get_x() + b.get_width() / 2, v + 0.3,
                 f"{v:.1f}", ha="center", fontsize=10)
    plt.tight_layout()
    plt.savefig("plot_profitability.png", dpi=150)
    plt.show()

    plt.figure(figsize=(8, 5))
    means = df.groupby("state")["plan_percent"].mean()
    bars = plt.bar(
        [state_names[s] for s in means.index],
        means.values,
        color=colors
    )
    plt.title("Среднее выполнение плана по состояниям")
    plt.xlabel("Состояние")
    plt.ylabel("Выполнение плана, %")
    plt.axhline(100, color="black", linestyle="--", linewidth=1, label="План = 100%")
    plt.legend()
    plt.grid(axis="y", alpha=0.4)
    for b, v in zip(bars, means.values):
        plt.text(b.get_x() + b.get_width() / 2, v + 0.5,
                 f"{v:.1f}", ha="center", fontsize=10)
    plt.tight_layout()
    plt.savefig("plot_plan.png", dpi=150)
    plt.show()

    plt.figure(figsize=(8, 5))
    means_over = df.groupby("state")["overdue_orders"].mean()
    means_def = df.groupby("state")["defects"].mean()
    x = np.arange(len(state_names))
    w = 0.35
    plt.bar(x - w / 2, means_over.values, w, label="Просроченные заказы", color="#1f77b4")
    plt.bar(x + w / 2, means_def.values, w, label="Дефекты", color="#9467bd")
    plt.xticks(x, [state_names[s] for s in means_over.index])
    plt.title("Операционные риски по состояниям")
    plt.xlabel("Состояние")
    plt.ylabel("Среднее количество")
    plt.legend()
    plt.grid(axis="y", alpha=0.4)
    plt.tight_layout()
    plt.savefig("plot_risks.png", dpi=150)
    plt.show()

    plt.figure(figsize=(11, 5))
    pivot = df.pivot_table(index="period", columns="state",
                           values="profitability", aggfunc="mean")
    for s in pivot.columns:
        plt.plot(pivot.index, pivot[s], marker="o", markersize=3,
                 label=state_names[s])
    plt.title("Динамика средней рентабельности по месяцам")
    plt.xlabel("Период")
    plt.ylabel("Рентабельность, %")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(alpha=0.4)
    plt.tight_layout()
    plt.savefig("plot_dynamics.png", dpi=150)
    plt.show()

    plt.figure(figsize=(7, 5))
    counts = df["state"].value_counts().sort_index()
    plt.pie(counts.values,
            labels=[state_names[s] for s in counts.index],
            autopct="%1.1f%%",
            colors=colors,
            startangle=90,
            wedgeprops={"edgecolor": "white"})
    plt.title("Доля наблюдений по состояниям")
    plt.tight_layout()
    plt.savefig("plot_distribution.png", dpi=150)
    plt.show()

    plt.figure(figsize=(9, 7))
    numeric = df[[
        "revenue", "costs", "employees", "productivity",
        "overdue_orders", "defects", "plan_percent",
        "profitability", "revenue_per_employee"
    ]]
    corr = numeric.corr()
    im = plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(im, shrink=0.8)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
    plt.yticks(range(len(corr.columns)), corr.columns)
    for i in range(len(corr)):
        for j in range(len(corr)):
            plt.text(j, i, f"{corr.iloc[i, j]:.2f}",
                     ha="center", va="center", fontsize=7,
                     color="white" if abs(corr.iloc[i, j]) > 0.5 else "black")
    plt.title("Корреляционная матрица показателей")
    plt.tight_layout()
    plt.savefig("plot_correlation.png", dpi=150)
    plt.show()