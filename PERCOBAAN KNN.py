import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# Load dataset dari file CSV
color_data = pd.read_csv('datasheet/colors.csv')
X = color_data[['R', 'G', 'B']].values
y = color_data['color_name'].values

# Normalisasi data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset untuk training dan testing
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Training model dengan KNN
knn = KNeighborsClassifier(n_neighbors=3)  # default k=3
knn.fit(X_train, y_train)

#prediksi
y_pred = knn.predict(X_test)

# Cek akurasi pada data test
accuracy = knn.score(X_test, y_test)
print(f"Akurasi Model KNN: {accuracy*100:.2f}%")

# Implementasi real-time dengan OpenCV
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Ambil pixel tengah gambar
    height, width, _ = frame.shape
    pixel_center = frame[height//2, width//2]
    pixel_center_scaled = scaler.transform([pixel_center])

    # Prediksi warna
    color_pred = knn.predict(pixel_center_scaled)[0]

    # Tampilkan hasil prediksi di layar
    cv2.putText(frame, f'Color: {color_pred}', (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
