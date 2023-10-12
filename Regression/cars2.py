# fmt: off
inputs = [
    (0.2, 1600), (1.0, 11000), (1.4, 23000), (1.6, 24000), (2.0, 30000), (2.2, 31000), (2.7, 35000), (2.8, 38000), (3.2, 40000), (3.3, 21000), (3.5, 45000), (3.7, 46000), (4.0, 50000), (4.4, 49000), (5.0, 60000), (5.2, 62000)]
    ## each input is a tuple of age and mileage
targets = [230, 555, 815, 860, 1140, 1085, 1200, 1330, 1290,
    870, 1545, 1480, 1750, 1845, 1790, 1955]
# fmt: on
w1 = 0.1
w2 = 0.2
b = 0.3
epochs = 4000  # massively increase the epochs
learning_rate = 1e-12  # bump the learning rate to essentially zero


def predict(i1: int | float, i2: int | float) -> int | float:
    return w1 * i1 + w2 * i2 + b


if __name__ == "__main__":
    # train the network
    for epoch in range(epochs):
        pred = [predict(i1, i2) for i1, i2 in inputs]
        cost = sum((p - t) ** 2 for p, t in zip(pred, targets)) / len(targets)
        print(f",epoch: {epoch} c: {cost:.2f}")

        # back-propagation
        errors_deriv = [2 * (p - t) for p, t in zip(pred, targets)]
        weight1_delta = [e * i[0] for e, i in zip(errors_deriv, inputs)]
        weight2_delta = [e * i[1] for e, i in zip(errors_deriv, inputs)]
        bias_delta = [e * 1 for e in errors_deriv]
        w1 -= learning_rate * sum(weight1_delta) / len(weight1_delta)
        w2 -= learning_rate * sum(weight2_delta) / len(weight2_delta)
        b -= learning_rate * sum(bias_delta) / len(bias_delta)

    print(f"w1:{w1:.4f}, w2:{w2:.4f}, b:{b:.4f}")
    # test the network (we expect a 1 year old car with 20000 miles to cost $750)
    print(f"Predicted cost: ${predict(1.0, 20000):.2f}")

# NB The network is mostly learning from the mileage and ignoring the weight due to differences in magnitude. Will rectify this later
