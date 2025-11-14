from laba import loading_model
from laba import make_model
from laba import model_training
from laba import visual_results
from laba import visual_accuracy
from laba import predict_from_file

def train_visual():
    x_train, y_train, x_test, y_test = loading_model()
    model = make_model()
    history = model_training(model, x_train, y_train)
    visual_results(history)
    visual_accuracy(history)
    return model, history, (x_test, y_test)

def main():
    model, history, (x_test, y_test) = train_visual()
    predict_from_file(model, 'number.png')

if __name__ == "__main__":
    main()