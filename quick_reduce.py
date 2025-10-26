import pandas as pd

# Leer y tomar las primeras 50,000 filas
df = pd.read_csv('data/creditcard.csv', nrows=50000)

print(f"📊 Primeras {len(df):,} filas tomadas")
print(f"💳 Fraudes: {(df['Class']==1).sum()}")
print(f"💳 Legítimas: {(df['Class']==0).sum()}")

# Dividir 80/20
from sklearn.model_selection import train_test_split

train, test = train_test_split(df, test_size=0.2, random_state=42)

# Guardar (sobreescribe los archivos grandes)
train.to_csv('data/train_data.csv', index=False)
test.to_csv('data/test_data.csv', index=False)

print(f"\n✅ Train: {len(train):,} filas guardado")
print(f"✅ Test: {len(test):,} filas guardado")