from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

csv_path = Path("datasets/predictions/deploy").resolve()

print(csv_path)

def get_csv(path):

    try:
        file_path = next(path.glob("*.csv"), None)
        df = pd.read_csv(filepath_or_buffer=file_path)
        print(df)
        return df

    except (ValueError, FileNotFoundError):
        print(f"Diretório ou arquivo não encontrado.")


df = get_csv(csv_path)

plt.figure()
plt.plot(df['date'], df['prediction'])

