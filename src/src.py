class Prediction():

    def __init__(self, theta0, theta1):
        self.theta0 = theta0
        self.theta1 = theta1

    def predict(self, mileage):
        return self.theta0 + mileage * self.theta1

    def training(self, lr, m, mileage_norm, price):
        tmp0 = lr / m * sum((self.predict(mileage_norm) - price))
        tmp1 = lr / m * \
            sum((self.predict(mileage_norm) - price) * mileage_norm)

        self.theta0 -= tmp0
        self.theta1 -= tmp1

    def scale(self, mileage_norm, mileage):
        price_lr = self.predict(mileage_norm)
        P1 = [mileage[0], mileage[6]]
        P2 = [price_lr[0], price_lr[6]]

        self.theta1 = (P2[1] - P2[0]) / (P1[1] - P1[0])
        self.theta0 = P2[0] - self.theta1 * P1[0]

    def save_model(self):
        filename = './model.csv'
        f = open(filename, 'w+')
        f.write(f'{self.theta0}, {self.theta1}')
        f.close()
