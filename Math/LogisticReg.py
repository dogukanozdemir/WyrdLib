def logistic_regression(x_train, y_train, x_test, y_test, learning_rate, num_iterations):

    import numpy as np

    def initialize_weights_and_bias(dimension):
        w = np.full((dimension, 1), 0.01)
        b = 0.0
        return w, b

    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def forward_backward_propagation(w, b, x_train, y_train):
        # forward propagation
        z = np.dot(w.T, x_train) + b
        y_head = sigmoid(z)
        loss = -y_train * np.log(y_head) - (1 - y_train) * np.log(1 - y_head)
        cost = (np.sum(loss)) / x_train.shape[1]

        # backward propagation
        derivative_weight = (np.dot(x_train, ((y_head - y_train).T))) / x_train.shape[1]
        derivative_bias = np.sum(y_head - y_train) / x_train.shape[1]
        gradients = {"derivative_weight": derivative_weight, "derivative_bias": derivative_bias}

        return cost, gradients

    def update(w, b, x_train, y_train, learning_rate, number_of_iterarion):
        cost_list = []

        for i in range(number_of_iterarion):
            cost, gradients = forward_backward_propagation(w, b, x_train, y_train)
            cost_list.append(cost)
            # lets update
            w = w - learning_rate * gradients["derivative_weight"]
            b = b - learning_rate * gradients["derivative_bias"]
            if i % 10 == 0:
                print ("Cost after iteration %i: %f" % (i, cost))

        # we update(learn) parameters weights and bias
        parameters = {"weight": w, "bias": b}
        return parameters, gradients, cost_list

    def predict(w, b, x_test):
        z = sigmoid(np.dot(w.T, x_test) + b)
        Y_prediction = np.zeros((1, x_test.shape[1]))
        # if z is bigger than 0.5, y_head=1
        # if z is smaller than 0.5, y_head=0
        for i in range(z.shape[1]):
            if z[0, i] <= 0.5:
                Y_prediction[0, i] = 0
            else:
                Y_prediction[0, i] = 1

        return Y_prediction

    # MAIN ##

    # initialize
    dimension = x_train.shape[0]
    w, b = initialize_weights_and_bias(dimension)

    parameters, gradients, cost_list = update(w, b, x_train, y_train, learning_rate, num_iterations)

    y_prediction_test = predict(parameters["weight"], parameters["bias"], x_test)

    accuracy = 100 - np.mean(np.abs(y_prediction_test - y_test)) * 100

    print("test accuracy: {} %".format(accuracy))
    
    return {"parameters" : parameters, "accuracy" : accuracy}