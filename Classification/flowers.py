import flowersdata as data
import math

weights = [[0.1, 0.2], [0.15, 0.25], [0.18, 0.1]]
biases = [0.3, 0.4, 0.35]
epochs = 100
learning_rate = 0.1


def softmax(predictions: float) -> list[float]:
    m = max(predictions)
    temp = [math.exp(p - m) for p in predictions]
    total = sum(temp)
    return [t / total for t in temp]


def log_loss(activations, targets):
    losses = [
        -t * math.log(a) - (1 - t) * math.log(1 - a)
        for a, t in zip(activations, targets)
    ]
    return sum(losses)


# train the network
for epoch in range(epochs):
    pred = [
        [sum(w * i for w, i in zip(we, inp)) + bi for we, bi in zip(weights, biases)]
        for inp in data.inputs
    ]
    # activation fn
    act = [softmax(p) for p in pred]
    cost = sum(log_loss(ac, ta) for ac, ta in zip(act, data.targets)) / len(act)

    # error and weight derivatives
    errors_d = [[(a - t) for a, t in zip(ac, ta)] for ac, ta in zip(act, data.targets)]
    inputs_T = list(zip(*data.inputs))  # transpose training inputs
    errors_d_T = list(zip(*errors_d))  # transpose error derivatives
    weights_d = [
        [sum(e * i for e, i in zip(er, inp)) for er in errors_d_T] for inp in inputs_T
    ]
    biases_d = [sum(errors) for errors in errors_d_T]

    weights_d_T = list(zip(*weights_d))  # transpose weight_deltas
    for y in range(len(weights_d_T)):
        for x in range(len(weights_d_T[0])):
            weights[y][x] -= learning_rate * weights_d_T[y][x] / len(data.inputs)
        biases[y] -= learning_rate * biases_d[y] / len(data.inputs)

    print(f"epoch: {epoch}, cost: {cost:.4f}")


# test the network
pred = [
    [sum(w * i for w, i in zip(we, inp)) + bi for we, bi in zip(weights, biases)]
    for inp in data.test_inputs
]
act = [softmax(p) for p in pred]
correct = sum(
    a.index(max(a)) == t.index(max(t)) for a, t in zip(act, data.test_targets)
)
print(f"Correct: {correct}/{len(act)} ({correct/len(act):.2%})")
