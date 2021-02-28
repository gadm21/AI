
import os
import torch



def mask_(matrices, maskval = 0.0, mask_diagonal = True):

    '''
    mask out all values in the given batch of matrices where i<=j holds,
    or i<j if mask_diagonal is false
    
    in-place operation
    '''

    b, h, w = matrices,size()

    indices = torch.triu_indices(h, w, offset = 0 if mask_diagonal else 1)
    matrices[:, indices[0], indices[1]] = maskval