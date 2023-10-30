import nonlineardata as data
import math


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


epochs = 1
learning_rate = 0.1

w_i_h = [[0.1, 0.2], [-0.3, 0.25], [0.12, 0.23], [-0.11, -0.22]]
w_h_o = [[0.2, 0.17, 0.3, -0.11], [0.3, -0.4, 0.5, -0.22], [0.12, 0.23, 0.15, 0.33]]
