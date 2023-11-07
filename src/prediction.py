import pandas
import matplotlib.pyplot as plt
import numpy as np
from src import Prediction


def visualizer(user_mileage, predicted_price):
    try:
        data = pandas.read_csv('./data.csv')
        mileage = np.array(data['km'])
        price = np.array(data['price'])

    except FileNotFoundError:
        print("No file data.csv, without dataset the visualizer can't work")
        exit()

    plt.title('Price of car based on mileage')
    plt.xlabel('Mileage (in miles)')
    plt.ylabel('Price (in $)')

    plt.plot(user_mileage, predicted_price, 'o', color='r')
    plt.plot(mileage, price, 'o', color='g')
    plt.show()


def main():

    filename = './model.csv'
    try:
        model_file = pandas.read_csv(filename, header=None)
        pr = Prediction(model_file.iloc[0, 0], model_file.iloc[0, 1])

    except FileNotFoundError:
        print("No file model.csv, the result will not be relevant")
        pr = Prediction(0, 0)

    mileage = input("What is the mileage of your car? ")
    while mileage.isnumeric() == False:
        print("Only numeric input")
        mileage = input("What is the mileage of your car? ")

    predicted_price = pr.predict(float(mileage))

    print("The price of your car is: " + str(predicted_price) + "$")

    visualizer(mileage, predicted_price)


if __name__ == "__main__":
    main()
