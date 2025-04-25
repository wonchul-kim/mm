
import numpy as np
import torch
import multiprocessing as mp
from mmengine.logging import MMLogger, print_log

def get_class_frequency(mask, num_classes):
    
    class_frequency = np.zeros(num_classes, dtype=np.int64)
    for class_id in range(num_classes):
        class_frequency[class_id] += np.sum(mask == class_id)
        
    return class_frequency

def get_class_frequency_v2(mask, num_classes):
    
    assert mask.ndim == 2, f"mask must be 2D, now ndim={mask.ndim}, shape={mask.shape}"

    flat_mask = mask.flatten()
    return np.bincount(flat_mask, minlength=num_classes)

def get_total_class_frequency(dataset, num_classes):
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.starmap(get_class_frequency_v2, [(batch['data_samples'].gt_sem_seg.data.numpy()[0], num_classes) for batch in dataset])
    
    total_freq = np.sum(results, axis=0)
    return total_freq


def get_class_weights(class_frequency, ignore_background=True):
    if ignore_background:
        class_frequency = class_frequency[1:]
    
    class_frequency = np.array(class_frequency, dtype=np.float64)
    weights = 1.0 / np.log1p(class_frequency)
    weights = len(class_frequency) * weights / np.sum(weights)

    if ignore_background == True:
        bg_weight = min(np.concatenate([[0.1], weights]))
        return np.concatenate([[bg_weight], weights])
    elif isinstance(ignore_background, float) and (1 >= ignore_background and ignore_background >= 0):
        return np.concatenate([[ignore_background], weights])
    else:
        return weights
    
def apply_class_weights_to_loss_decode(runner, class_weights):
    logger: MMLogger = MMLogger.get_current_instance()

    if hasattr(runner.model.decode_head, 'loss_decode'):
        if isinstance(runner.model.decode_head.loss_decode, (list, tuple, torch.nn.modules.container.ModuleList)):
            for loss_decode in runner.model.decode_head.loss_decode:
                if hasattr(loss_decode, 'class_weight'):
                    loss_decode.class_weight = class_weights
                    print_log(f"APPLIED class-weights ({class_weights}) to model.decode_head.loss_decode ({loss_decode})", logger)
        else:
            if hasattr(runner.model.decode_head.loss_decode, 'class_weight'):
                runner.model.decode_head.loss_decode.class_weight = class_weights
                print_log(f"APPLIED class-weights ({class_weights}) to model.decode_head.loss_decode ({loss_decode})", logger)

def change_class_weights_to_loss_decode(runner, class_weights):
    logger: MMLogger = MMLogger.get_current_instance()

    def _change_class_weights(class_weights, loss_decode):
        if len(class_weights) == 1:
            loss_decode.class_weight[0] = class_weights[0]
        elif isinstance(class_weights, (int, float)):
            loss_decode.class_weight[0] = class_weights
        elif len(class_weights) == len(loss_decode.class_weight):
            loss_decode.class_weight = class_weights
        else:
            NotImplementedError(f'NOT Considered this case of class-weight to change class-weights: {class_weights}')
    
    if hasattr(runner.model.decode_head, 'loss_decode'):
        if isinstance(runner.model.decode_head.loss_decode, (list, tuple, torch.nn.modules.container.ModuleList)):
            for loss_decode in runner.model.decode_head.loss_decode:
                if hasattr(loss_decode, 'class_weight'):
                    _change_class_weights(class_weights, loss_decode)
                    print_log(f"CHANGED class-weights ({class_weights}) to model.decode_head.loss_decode ({loss_decode})", logger)
        else:
            if hasattr(loss_decode, 'class_weight'):
                _change_class_weights(class_weights, loss_decode)
                print_log(f"CHANGED class-weights ({class_weights}) to model.decode_head.loss_decode ({loss_decode})", logger)
        
    
if __name__ == '__main__':

    mask = np.zeros((1120, 768, 4))
    mask[0:20, 20:30, 1] = 1
    mask[0:20, 20:30, 2] = 2
    mask[0:20, 20:30, 3] = 3

    class_frequency = get_class_frequency(mask, 4)
    print(class_frequency)
    class_weights = get_class_weights(class_frequency)
    print(class_weights)