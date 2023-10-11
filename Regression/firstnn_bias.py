w = 0.1
b = 0.3
learning_rate = 0.1
epochs = 100

inputs = [1, 2, 3, 4]
targets = [12, 14, 16, 18]


def predict(input: int | float) -> int | float:
    return w * input + b


if __name__ == "__main__":
    # train the network
    for _ in range(epochs):
        pred = [predict(i) for i in inputs]
        errors = [(t - p) ** 2 for p, t in zip(pred, targets)]
        cost = sum(errors) / len(targets)
        print(f"Weight: {w:.2f}, Bias:{b:.2f}, Cost: {cost:.2f}")

        # take the derivative of the error function
        errors_deriv = [2 * (p - t) for p, t in zip(pred, targets)]
        weight_delta = [e * i for e, i in zip(errors_deriv, inputs)]
        bias_delta = [e * 1 for e in errors_deriv]
        w -= learning_rate * sum(weight_delta) / len(weight_delta)
        b -= learning_rate * sum(bias_delta) / len(bias_delta)

    # test the network
    test_inputs = [5, 6]
    test_targets = [20, 22]
    pred = [predict(i) for i in test_inputs]
    print("\nTest Data:")
    for i, t, p in zip(test_inputs, test_targets, pred):
        print(f"input: {i}, target: {t}, pred: {p:.4f}")
