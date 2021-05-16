
import os
import torch



def mask_(matrices, maskval = 0.0, mask_diagonal = True):

    '''
    mask out all values in the given batch of matrices where i<=j holds,
    or i<j if mask_diagonal is false
    
    in-place operation
    '''

    b, h, w = matrices.size()

    indices = torch.triu_indices(h, w, offset = 0 if mask_diagonal else 1)
    matrices[:, indices[0], indices[1]] = maskval







def enwik(path, n_train=int(90e6), n_valid=int(5e6), n_test=int(5e6)):

  with open(path) as file : 
    str_data = file.read(n_train+ n_valid+ n_test)
    data = np.fromstring(str_data, dtype = np.uint8)
    
    train_x, val_x, test_x = np.split(data, [n_train, n_train+ n_valid]) 
    return torch.from_numpy(train_x), torch.from_numpy(val_x), torch.from_numpy(test_x)
