# main.py
# консольное меню: генерация датасета, графики, оценка поставщиков

from fuzzy_sets import evaluate, verdict
from data_utils import generate_data
from inference import plot_membership, plot_distribution


TEST_SUPPLIERS = [
    ("Поставщик A (отличный)",  2,  1, 99,  0,  3),
    ("Поставщик B (хороший)",   6,  3, 95,  2, 10),
    ("Поставщик C (средний)",  12,  8, 88, -3, 25),
    ("Поставщик D (слабый)",   20, 12, 80,  8, 40),
    ("Поставщик E (плохой)",   28, 18, 72, 14, 55),
]


def gen_dataset():
    df = generate_data()
    print(f"\nСгенерировано записей: {len(df)}")
    print(df.head())
    df.to_csv("suppliers.csv", index=False)
    print("Сохранил в suppliers.csv")


def show_plots():
    print("\nСтрою графики функций принадлежности...")
    plot_membership()
    print("Строю гистограмму распределения индекса...")
    plot_distribution()
    print("Готово. Сохранены membership_functions.png и reliability_distribution.png")


def check_tests():
    print("\nПРОВЕРКА НА ТЕСТОВЫХ ПОСТАВЩИКАХ")
    for name, dd, qd, comp, pdev, payd in TEST_SUPPLIERS:
        idx, (ml, mm, mh) = evaluate(dd, qd, comp, pdev, payd)
        print(f"\n{name}")
        print(f"  Индекс надежности: {idx:.2f}")
        print(f"  mu(низкая):  {ml:.3f}")
        print(f"  mu(средняя): {mm:.3f}")
        print(f"  mu(высокая): {mh:.3f}")
        print(f"  Вердикт: {verdict(idx)} надежность")


def ask_supplier():
    print("\nВвод данных нового поставщика")
    try:
        dd   = float(input("Средняя задержка поставки (0-30 дней): "))
        qd   = float(input("Доля дефектной продукции (0-20 %): "))
        comp = float(input("Полнота поставки (70-100 %): "))
        pdev = float(input("Отклонение цены (-15..15 %): "))
        payd = float(input("Просрочка оплаты (0-60 дней): "))
    except ValueError:
        print("Ошибка: вводите только числа.")
        return

    idx, (ml, mm, mh) = evaluate(dd, qd, comp, pdev, payd)
    print(f"\n  Индекс надежности: {idx:.2f}")
    print(f"  mu(низкая):  {ml:.3f}")
    print(f"  mu(средняя): {mm:.3f}")
    print(f"  mu(высокая): {mh:.3f}")
    print(f"  Итог: {verdict(idx)} надежность")


def menu():
    while True:
        print("\n" + "-" * 50)
        print("  ОЦЕНКА НАДЕЖНОСТИ ПОСТАВЩИКА")
        print("-" * 50)
        print("1 - Сгенерировать датасет (850 поставщиков)")
        print("2 - Построить графики")
        print("3 - Оценить нового поставщика")
        print("4 - Показать тестовых поставщиков")
        print("5 - Выход")
        choice = input("Выберите пункт: ").strip()

        if choice == "1":
            gen_dataset()
        elif choice == "2":
            show_plots()
        elif choice == "3":
            ask_supplier()
        elif choice == "4":
            check_tests()
        elif choice == "5":
            print("Пока!")
            break
        else:
            print("Нет такого пункта, попробуйте снова.")


if __name__ == "__main__":
    menu()