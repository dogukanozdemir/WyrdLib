class LinearReg:
    def __init__(self, x, y):
        if len(x) != len(y):
            raise ValueError("Both axis must have the same number of values")

        self.x = x
        self.y = y
        self.x_mean = self.axis_mean(self.x)
        self.y_mean = self.axis_mean(self.y)
        self.b_zero = 0

    @staticmethod
    def axis_mean(axis):
        return sum(axis) / len(axis)

    def sum_of_deviation_products(self):
        sum((x - self.x_mean) * (y - self.y_mean) for x, y in zip(self.x, self.y))

    def sum_of_x_deviation_squared(self):
        sum((x - self.x_mean) ** 2 for x in self.x)

    @property
    def slope(self):
        return self.sum_of_deviation_products() / self.sum_of_x_deviation_squared()

    def fit_best_line(self):
        self.b_zero = self.y_mean - (self.slope * self.x_mean)
        print(f"The best line equation is: {self.slope:.2f}x + {self.b_zero:.2f}")