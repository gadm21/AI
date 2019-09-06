import math

'''
create batches of features and labels of batch_size size
:param batch_size: max size of each batch (the last batch might be of size less than batch_size)
:param features: list of all features to extract batches from
:param labels: list of all labels to extract batches from
:return : Batches of (features, labels) of size batch_size at most.
'''
def get_batches(batch_size, features, labels):
    
    assert len(features)== len(labels)
    
    batches= []
    
    #total stamples to batch to pieces
    sample_size= len(features)
    for start_i in range(0, sample_size, batch_size):  #here, the step size is the batch_size
        end_i= start_i+ batch_size
        batch= [features[start_i: end_i], labels[start_i: end_i]]
        batches.append(batch)
    
    return batches