import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# 1. Baca dataset
df = pd.read_csv('diabetes.csv')

# 2. Pisahkan fitur (X) dan target/label (y)
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# 3. Split data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Standarisasi (Scaling) Data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# 5. Latih Model Machine Learning
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# 6. Simpan model dan scaler ke dalam format .pkl
pickle.dump(scaler, open('scaler.pkl', 'wb'))
pickle.dump(model, open('model.pkl', 'wb'))

print("Berhasil! File scaler.pkl dan model.pkl sudah dibuat di folder kamu.")