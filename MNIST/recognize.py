import os
import numpy as np
from PIL import Image
from tensorflow import keras

# ==== НАСТРОЙКИ ====
MODEL_PATH = 'mnist_model.keras'
DIGITS_FOLDER = 'digits'
# ====================

def load_model():
    """Загружаем обученную модель"""
    print(f"Загружаю модель из {MODEL_PATH}...")
    model = keras.models.load_model(MODEL_PATH)
    print("Модель загружена!\n")
    return model


def preprocess_image(image_path):
    """
    Подготавливаем изображение для модели:
    - конвертируем в градации серого
    - изменяем размер до 28x28
    - нормализуем значения пикселей
    - инвертируем цвета (в MNIST фон черный, цифра белая)
    """
    img = Image.open(image_path).convert('L')  # градации серого
    img = img.resize((28, 28))
    
    img_array = np.array(img).astype('float32') / 255.0
    
    # Если у вас фон белый, а цифра черная - нужна инверсия
    # Если наоборот (как в MNIST) - закомментируйте следующую строку
    img_array = 1 - img_array
    
    img_array = img_array.reshape(1, 28, 28, 1)
    return img_array


def predict_digit(model, image_path):
    """Предсказываем цифру на изображении"""
    img_array = preprocess_image(image_path)
    prediction = model.predict(img_array, verbose=0)
    
    digit = np.argmax(prediction)
    confidence = np.max(prediction) * 100
    
    return digit, confidence


def process_folder(model, folder_path):
    """Обрабатываем все PNG файлы в папке"""
    
    if not os.path.exists(folder_path):
        print(f"Ошибка: папка '{folder_path}' не найдена!")
        return
    
    # Собираем все PNG файлы
    png_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.png')]
    
    if not png_files:
        print(f"В папке '{folder_path}' не найдено PNG файлов.")
        print(f"Положите туда картинки с цифрами и запустите скрипт снова.")
        return
    
    print(f"Найдено файлов: {len(png_files)}\n")
    print("=" * 50)
    
    for filename in sorted(png_files):
        file_path = os.path.join(folder_path, filename)
        digit, confidence = predict_digit(model, file_path)
        
        print(f"📄 Файл: {filename}")
        print(f"   Предсказанная цифра: {digit}")
        print(f"   Уверенность: {confidence:.1f}%")
        print("-" * 50)


def main():
    model = load_model()
    process_folder(model, DIGITS_FOLDER)


if __name__ == "__main__":
    main()