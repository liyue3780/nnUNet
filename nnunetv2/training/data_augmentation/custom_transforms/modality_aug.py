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


# ------- an augmentation where we simulate one of the modalities having partial coverage -------
def augment_partial_coverage(data_dict, b) -> None:
    # The direction of augmentation is not obvious to me, but empirically it seems that if we 
    # encode the input images to be RPI, dimension 0 ends up being the correct (anterior-posterior)
    # direction by the time we get here
    augdir = 0
    
    # Get the shape of the image patch
    patch = data_dict['data'][b]
    patch_shape = patch.shape
    
    # The fade will have the format 
    #   f(t) = 0 for t < t0-w, f(t) = 0.5 + 0.5 * (t-t0) / w for t0-w <= t <= t0+w, and f(t) = 1 for t > t0+w 
    # where t is the coordinate along the augdir direction, t0 is the center of the fade, and w is the width of the fade. 
    # The values in the fade will be multiplied to the image to simulate partial coverage.
    
    # Determine the direction of the fade (towards anterior or posterior), at random
    fade_dir = -1 if np.random.uniform() < 0.5 else 1
    
    # Determine the percentage of the volume that is affected, at random
    t0 = np.random.uniform(0.0, 1.0)
    
    # The width of the fade should be specified in mm, we can convert to voxel units using spacing
    w = np.random.uniform(0.5, 1.5) / (data_dict['properties'][b]['spacing'][augdir] * patch_shape[augdir+1])

    # Create the fade
    t = np.linspace(0.0, 1.0, data_dict['data'][0].shape[augdir+1])
    fade = 0.5 + fade_dir * (np.clip((t - t0) / w, 0.0, 1.0) - 0.5)
    
    # Apply the fade to a single randomly chosen modality (TODO: should we handle more modalities?)
    cn_ = random.randint(0, patch_shape[0]-1)

    # Choose the value to use for the fade - don't just fix to zero, but sample randomly from image range
    value = np.random.uniform(patch[cn_,:,:,:].min(), patch[cn_,:,:,:].max())
    
    reshape_vec = [-1 if (i-1) == augdir else 1 for i in range(len(patch_shape))]
    new_array = data_dict['data'][b][cn_] * fade.reshape(reshape_vec) + value * (1 - fade.reshape(reshape_vec))
    
    data_dict['data'][b][cn_] = new_array


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


class ModalityAugAndPartialCoverageTransform(AbstractTransform):
    def __init__(self, data_key="data") -> None:
        super().__init__()
        self.data_key = data_key
    
    def __call__(self, **data_dict):
        for b in range(len(data_dict[self.data_key])):
            # Select type of augmentation to apply
            randval = np.random.uniform()
            if randval < 0.4:
                log_verbose('do not implement the modality augmentation')
            elif randval < 0.7:
                data_dict[self.data_key][b] = augment_missed_modality_any_number(data_dict[self.data_key][b])
            else:
                augment_partial_coverage(data_dict, b)

        return data_dict
      
      
# ------- work for first two modalities -------
def augment_missed_modality_only_for_first_two_channels(data_sample: np.ndarray):

    channel_num = data_sample.shape[0]
    if channel_num > 1:
        missing_num = random.randint(1, 1)  # only augment one of the first two channels
        aug_cn = random.sample(range(2), missing_num)
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


class ModalityAugFirstTwoChannelsTransform(AbstractTransform):
    def __init__(self, data_key="data") -> None:
        super().__init__()
        self.data_key = data_key
    
    def __call__(self, **data_dict):
        for b in range(len(data_dict[self.data_key])):
            if np.random.uniform() < 0.5:
                data_dict[self.data_key][b] = augment_missed_modality_only_for_first_two_channels(data_dict[self.data_key][b])
            else:
                log_verbose('do not implement the modality augmentation')

        return data_dict