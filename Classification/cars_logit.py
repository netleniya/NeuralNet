import math

# fmt: off
inputs = [
    (0.0000, 0.0000), (0.1600, 0.1556), (0.2400, 0.3543),
    (0.2800, 0.3709),(0.3600, 0.4702), (0.4000, 0.4868),
    (0.5000, 0.5530), (0.5200, 0.6026),(0.6000, 0.6358),
    (0.6200, 0.3212), (0.6600, 0.7185), (0.7000, 0.7351),
    (0.7600, 0.8013), (0.8400, 0.7848), (0.9600, 0.9669),
    (1.0000, 1.0000)]
    ## each input is a tuple of normalized age and mileage
targets = [0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1] # 0 = keep, 1 = sell
# fmt: on
weights = [0.1, 0.2]
b = 0.3
epochs = 4000  # massively increase the epochs
learning_rate = 0.1


def predict(inputs: list) -> int | float:
    return sum(w * i for w, i in zip(weights, inputs)) + b


def activate(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def log_loss(act: float, target: int) -> float:
    return -target * math.log(act) - (1 - target) * math.log(1 - act)


if __name__ == "__main__":
    # train the network
    for epoch in range(epochs):
        pred = [predict(input) for input in inputs]
        act = [activate(p) for p in pred]
        cost = sum(log_loss(a, t) for a, t in zip(act, targets)) / len(act)
        print(f",epoch: {epoch} c: {cost:.2f}")

        # back-propagation
        errors_deriv = [(a - t) for a, t in zip(act, targets)]
        weights_d = [
            [err * i for i in input] for err, input in zip(errors_deriv, inputs)
        ]
        bias_delta = [e * 1 for e in errors_deriv]
        weights_d_T = list(zip(*weights_d))  # transpose weights_d
        for i in range(len(weights)):
            weights[i] -= learning_rate * sum(weights_d_T[i]) / len(weights_d)
        b -= learning_rate * sum(bias_delta) / len(bias_delta)

    # test the network with normalized test data
    test_inputs = [
        (0.1600, 0.1391),
        (0.5600, 0.3046),
        (0.7600, 0.8013),
        (0.9600, 0.3046),
        (0.1600, 0.7185),
    ]
    test_targets = [0, 0, 1, 0, 0]

    pred = [predict(input) for input in test_inputs]
    act = [activate(p) for p in pred]
    for a, t in zip(act, test_targets):
        print(f"target: {t}, predicted: {a:.0f}")
