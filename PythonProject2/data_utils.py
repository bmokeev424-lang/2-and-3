import numpy as np
import pandas as pd

def generate_data(n=850, seed=42):
    np.random.seed(seed)
    df = pd.DataFrame({
        'delivery_delay':  np.random.uniform(0, 30, n),
        'quality_defects': np.random.uniform(0, 20, n),
        'completeness':    np.random.uniform(70, 100, n),
        'price_deviation': np.random.uniform(-15, 15, n),
        'payment_delay':   np.random.uniform(0, 60, n),
    })
    return df

if __name__ == "__main__":
    df = generate_data()
    print("Записей:", len(df))
    print(df.head())
    df.to_csv("suppliers.csv", index=False)
    print("Сохранил в suppliers.csv")