from batchgenerators.transforms.abstract_transforms import AbstractTransform
import numpy as np
import random
import os

verbose_ = os.environ.get('ModAug_VERBOSE', '0') == '1'

def log_verbose(msg):
    if verbose_:
        print(msg)


def augment_missed_modality_all_four(data_sample: np.ndarray):
    channel_num = data_sample.shape[0]
    if channel_num != 5:
        raise ValueError('Augmentation for all four modalities requires input data has five channels')
    
    # select the augmentation method
    missing_num = random.randint(1, 4)
    aug_cn = random.sample(range(channel_num), missing_num)
    
    log_verbose('modality augmentation: imitate missing {} modality/modalities: {}'.format(len(aug_cn), str(aug_cn)))

    for cn_ in aug_cn:
        new_array = np.random.randn(data_sample[cn_].shape[0],\
                                    data_sample[cn_].shape[1],\
                                    data_sample[cn_].shape[2])
        data_sample[cn_] = new_array
    
    return data_sample


class ModalityAugAllFourTransform(AbstractTransform):
    def __init__(self, data_key="data") -> None:
        super().__init__()
        self.data_key = data_key
    
    def __call__(self, **data_dict):
        for b in range(len(data_dict[self.data_key])):
            if np.random.uniform() < 0.5:
                data_dict[self.data_key][b] = augment_missed_modality_all_four(data_dict[self.data_key][b])
            else:
                log_verbose('do not implement the modality augmentation')

        return data_dict


# ------- work for any modalities -------
def augment_missed_modality_any_number(data_sample: np.ndarray):
    channel_num = data_sample.shape[0]

    if channel_num > 1:
        missing_num = random.randint(1, channel_num-1)
        aug_cn = random.sample(range(channel_num), missing_num)

    elif channel_num == 1:
        log_verbose('No channel can be augmented')
        return data_sample
        
    else:
        raise
    
    log_verbose('modality augmentation: imitate missing {} modality/modalities: {}'.format(len(aug_cn), str(aug_cn)))

    for cn_ in aug_cn:
        if len(data_sample[cn_].shape) == 3:
            # for 3D
            new_array = np.random.randn(data_sample[cn_].shape[0], data_sample[cn_].shape[1], data_sample[cn_].shape[2])
            data_sample[cn_] = new_array

        elif len(data_sample[cn_].shape) == 2:
            # for 2D
            new_array = np.random.randn(data_sample[cn_].shape[0], data_sample[cn_].shape[1])
            data_sample[cn_] = new_array
    
    return data_sample


class ModalityAugAnyNumberTransform(AbstractTransform):
    def __init__(self, data_key="data") -> None:
        super().__init__()
        self.data_key = data_key
    
    def __call__(self, **data_dict):
        for b in range(len(data_dict[self.data_key])):
            if np.random.uniform() < 0.5:
                data_dict[self.data_key][b] = augment_missed_modality_any_number(data_dict[self.data_key][b])
            else:
                log_verbose('do not implement the modality augmentation')

        return data_dict