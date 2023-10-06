import numpy as np
import tensorflow as tf
from tensorflow import keras

# Задаем начальные параметры
np.random.seed(25)
tf.random.set_seed(25)

# Генерируем наше число
target_number = np.random.randint(1, 101)

# Генерируйте обучающие данные
def generate_training_data(num_samples):
    data = []
    for _ in range(num_samples):
        attempts = np.random.randint(1, 21)  # Random attempts between 1 and 20
        feedback = 1 if target_number > attempts else 0
        data.append((attempts, feedback))
    return data

# Создаем простейшую нейронную сеть
def create_model():
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(1,)),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Обучаем неронную сеть
def train_model(model, data):
    X = np.array([attempts for attempts, _ in data])
    y = np.array([feedback for _, feedback in data])
    model.fit(X, y, epochs=10, batch_size=32)

# Создаем предсказания
def make_prediction(model, attempts):
    input_data = np.array([attempts])
    predicted_probability = model.predict(input_data)[0][0]
    return predicted_probability

# Используем алгоритм бинарного поиска
def binary_search(target_number):
    lower_bound = 1
    upper_bound = 100
    attempts = 0

    while lower_bound <= upper_bound:
        mid = (lower_bound + upper_bound) // 2
        attempts += 1

        if mid == target_number:
            return attempts
        elif mid < target_number:
            lower_bound = mid + 1
        else:
            upper_bound = mid - 1

    return attempts

# Генерируем данные для обучения
training_data = generate_training_data(10000)
model = create_model()
train_model(model, training_data)

# Попытка угадать число нейронной сетью
nn_attempts = 0
while nn_attempts < 20:
    nn_attempts += 1
    predicted_probability = make_prediction(model, nn_attempts)

    if predicted_probability >= 0.5:
        break

# Попытка угадать число бинарным поиском
bs_attempts = binary_search(target_number)

# Сравниваем результаты
print(f"Neural Network: {nn_attempts} attempts")
print(f"Binary Search: {bs_attempts} attempts")