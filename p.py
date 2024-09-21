import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import concurrent.futures
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Generate or load data (contoh data digits dari sklearn)
digits = load_digits()
X, y = digits.data, digits.target

# Visualisasi dataset dengan Matplotlib
plt.figure(figsize=(10, 5))
for i in range(1, 11):
    plt.subplot(2, 5, i)
    plt.imshow(digits.images[i], cmap='gray')
    plt.title(f"Label: {y[i]}")
    plt.axis('off')
plt.show()

# Bagi dataset menjadi training dan testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Buat fungsi untuk training model menggunakan RandomForest secara paralel
def train_model(n_estimators):
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return f"Model dengan {n_estimators} trees, Akurasi: {accuracy:.2f}"

# Lakukan training secara paralel dengan berbagai parameter
with concurrent.futures.ThreadPoolExecutor() as executor:
    results = executor.map(train_model, [10, 50, 100, 200, 500])

# Print hasil
for result in results:
    print(result)

# Visualisasi salah satu prediksi
sample_idx = np.random.choice(len(X_test), 5)
plt.figure(figsize=(10, 5))
for i, idx in enumerate(sample_idx, 1):
    plt.subplot(1, 5, i)
    plt.imshow(X_test[idx].reshape(8, 8), cmap='gray')
    plt.title(f"Pred: {clf.predict([X_test[idx]])[0]}\nTrue: {y_test[idx]}")
    plt.axis('off')
plt.show()
