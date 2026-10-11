import matplotlib.pyplot as plt
from fuzzy_sets import (
    delivery_delay, quality_defects, completeness,
    price_deviation, payment_delay, reliability, evaluate
)
from data_utils import generate_data


def plot_membership():
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    pairs = [
        (delivery_delay,  'delivery_delay',  ['low', 'medium', 'high']),
        (quality_defects, 'quality_defects', ['low', 'medium', 'high']),
        (completeness,    'completeness',    ['low', 'medium', 'high']),
        (price_deviation, 'price_deviation', ['negative', 'neutral', 'positive']),
        (payment_delay,   'payment_delay',   ['low', 'medium', 'high']),
        (reliability,     'reliability',     ['low', 'medium', 'high']),
    ]

    for ax, (var, title, terms) in zip(axes.flat, pairs):
        for t in terms:
            ax.plot(var.universe, var[t].mf, label=t, linewidth=2)
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("membership_functions.png", dpi=120)
    plt.show()


def plot_distribution():
    df = generate_data()
    indices = []
    for _, row in df.iterrows():
        idx, _ = evaluate(row['delivery_delay'], row['quality_defects'],
                          row['completeness'], row['price_deviation'],
                          row['payment_delay'])
        indices.append(idx)

    avg = sum(indices) / len(indices)
    plt.figure(figsize=(9, 5))
    plt.hist(indices, bins=40, color='steelblue', edgecolor='black', alpha=0.8)
    plt.axvline(avg, color='red', linestyle='--', label=f'Среднее = {avg:.1f}')
    plt.xlabel('Индекс надежности (0–100)')
    plt.ylabel('Количество поставщиков')
    plt.title('Распределение итогового индекса надежности')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("reliability_distribution.png", dpi=120)
    plt.show()


if __name__ == "__main__":
    plot_membership()
    plot_distribution()