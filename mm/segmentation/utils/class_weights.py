
import numpy as np

def get_class_frequency(mask, num_classes):
    
    class_frequency = np.zeros(num_classes, dtype=np.int64)
    for class_id in range(num_classes):
        class_frequency[class_id] += np.sum(mask == class_id)
        
    return class_frequency

def get_class_weights(class_frequency, ignore_background=True):
    if ignore_background:
        class_frequency = class_frequency[1:]
    
    class_frequency = np.array(class_frequency, dtype=np.float64)
    weights = 1.0 / np.log1p(class_frequency)
    weights = len(class_frequency) * weights / np.sum(weights)

    if ignore_background:
        return np.concatenate([[0], weights])
    else:
        return weights
    
if __name__ == '__main__':

    mask = np.zeros((1120, 768, 4))
    mask[0:20, 20:30, 1] = 1
    mask[0:20, 20:30, 2] = 2
    mask[0:20, 20:30, 3] = 3

    class_frequency = get_class_frequency(mask, 4)
    print(class_frequency)
    class_weights = get_class_weights(class_frequency)
    print(class_weights)