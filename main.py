from lab import train_visual
from lab import predict_digit_from_file

def main():
    
    model, history, (x_test, y_test) = train_visual()
    predict_digit_from_file(model, 'number.png')

if __name__ == "__main__":
    main()