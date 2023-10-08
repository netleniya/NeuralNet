w = 0.1
learning_rate = 0.1

inputs = [1, 2, 3, 4]
targets = [2, 4, 6, 8]


def predict(input: int | float) -> int | float:
    return w * input


if __name__ == "__main__":
    # train the network
    for _ in range(25):
        pred = [predict(i) for i in inputs]
        errors = [(t - p) for p, t in zip(pred, targets)]
        cost = sum(errors) / len(targets)
        print(f"Weight: {w:.2f}, Cost: {cost:.2f}")
        w += learning_rate * cost

    # test the network
    test_inputs = [5, 6]
    test_targets = [10, 12]
    pred = [predict(i) for i in test_inputs]
    for i, t, p in zip(test_inputs, test_targets, pred):
        print(f"input: {i}, target: {t}, pred: {p:.4f}")
