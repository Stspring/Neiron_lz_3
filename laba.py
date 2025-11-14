from keras.datasets import mnist
from keras import models, layers
from keras.utils import to_categorical
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def loading_model():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()        
    x_train = x_train.reshape((60000, 28 * 28)).astype("float32") / 255     
    x_test = x_test.reshape((10000, 28 * 28)).astype("float32") / 255
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    return x_train, y_train, x_test, y_test

def make_model():
    model = models.Sequential([
        layers.Dense(512, activation='relu', input_shape=(28 * 28,)),   # Скрытый слой.
        layers.Dropout(0.3),
        layers.Dense(10, activation='softmax')                          #  Входной слой.
    ])
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def model_training(model, x_train, y_train):        # Обучаем модель.
    history = model.fit(
        x_train,
        y_train,
        epochs=5,
        batch_size=128,
        validation_split=0.1
    )
    return history

def visual_results(history):                                    # Строим график для анализа модели (процесса обучения).
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title('Loss during training')
    plt.legend()
    plt.show()

def visual_accuracy(history):                                   # Строим график для анализа точности.
    plt.plot(history.history['accuracy'], label='train_accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.title('Accuracy during training')
    plt.legend()
    plt.show()


def predict_from_file(model, filename):
    img = Image.open(filename).convert('L')         # Загружаем изображение и приводим к формату MNIST.
    img = img.resize((28, 28))

    plt.imshow(img, cmap='gray')                    # Визуализируем картинку, для проверки корректной загрузки.
    plt.title('Загруженное изображение')
    plt.axis('off')
    plt.show()

    img_array = np.array(img)                       # Преобразуем в numpy.

    # Инвертируем значения, если фон светлый,
    # а цифра тёмная — приводим к стилю MNIST.
    if np.mean(img_array) > 127:
        img_array = 255 - img_array



    img_array = img_array.astype('float32') / 255           # Нормализуем.

    prediction = model.predict(img_array.reshape(1, 28 * 28))
    predicted_class = np.argmax(prediction)
    print(f'Предсказанная цифра: {predicted_class}')



# ---------------------------------------------
# ОБЪЕДИНЁННАЯ ФУНКЦИЯ (если нужно запускать всё разом)
# ---------------------------------------------
def train_visual():
    x_train, y_train, x_test, y_test = prepare_mnist()
    model = make_network()
    history = run_training(model, x_train, y_train)
    show_loss(history)
    show_accuracy(history)
    return model, history, (x_test, y_test)
