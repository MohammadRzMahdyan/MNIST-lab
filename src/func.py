from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.metrics import Accuracy
import numpy as np

INPUT_SIZE = 28 * 28
NUM_CLASSES = 10

def load_mnist(path):
    data = np.load(path)
    X_train = data['x_train']
    y_train = data['y_train']
    X_test = data['x_test']
    y_test = data['y_test']
    return (X_train, y_train), (X_test, y_test)
    
    
def build_model():
    model = Sequential([
        Input(shape=(INPUT_SIZE,)),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    return model
    
def compile_model(model):
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return
    
def train(model, X, y):
    return model.fit(X, y)
    
def evaluate(model, X, y):
    _, acc = model.evaluate(X, y)
    return acc
    
def predict_classes(model, X):
    probs = model.predict(X)
    return probs.argmax(axis=1)
    
    
def preprocess_images(X):
    return X.reshape(-1, INPUT_SIZE) / 255.00
