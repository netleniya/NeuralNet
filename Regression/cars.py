# fmt: off
inputs = [0.2, 1.0, 1.4, 1.6, 2.0, 2.2, 2.7, 2.8, 3.2,
    3.3, 3.5, 3.7, 4.0, 4.4, 5.0, 5.2]
targets = [230, 555, 815, 860, 1140, 1085, 1200, 1330, 1290,
    870, 1545, 1480, 1750, 1845, 1790, 1955]
# fmt: on
w = 0.1
b = 0.3
epochs = 200
learning_rate = 0.01


def predict(input: int | float) -> int | float:
    return w * input + b


if __name__ == "__main__":
    # train the network
    for epoch in range(epochs):
        pred = [predict(i) for i in inputs]
        cost = sum([(p - t) ** 2 for p, t in zip(pred, targets)]) / len(targets)
        print(f"w:{w:.2f}, b:{b:.2f}, c: {cost:.2f}")

        # back-propagation
        errors_deriv = [2 * (p - t) for p, t in zip(pred, targets)]
        weight_delta = [e * i for e, i in zip(errors_deriv, inputs)]
        bias_delta = [e * 1 for e in errors_deriv]
        w -= learning_rate * sum(weight_delta) / len(weight_delta)
        b -= learning_rate * sum(bias_delta) / len(bias_delta)
