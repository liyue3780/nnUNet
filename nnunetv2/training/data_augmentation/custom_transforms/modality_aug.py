from batchgenerators.transforms.abstract_transforms import AbstractTransform
import numpy as np
import random


def augment_missed_modality_all_four(data_sample: np.ndarray):
    channel_num = data_sample.shape[0]
    if channel_num != 5:
        raise ValueError('Augmentation for all four modalities requires input data has five channels')
    
    # select the augmentation method
    missing_num = random.randint(1, 4)
    aug_cn = random.sample(range(channel_num), missing_num)
    print('modality augmentation: imitate missing {} modality/modalities: {}'.format(len(aug_cn), str(aug_cn)))

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
                print('do not implement the modality augmentation')

        return data_dict