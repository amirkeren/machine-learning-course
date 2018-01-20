def get_batch(X, y, current_batch_index, batch_size):
    batch_start_index = current_batch_index * batch_size
    batch_end_index = (current_batch_index + 1) * batch_size
    batch_features = X[batch_start_index:batch_end_index]
    batch_labels = y[batch_start_index:batch_end_index]
    return batch_features, batch_labels


def mean_squared_error(y1, y2):
    import numpy as np

    return np.mean((y1 - y2) ** 2)
