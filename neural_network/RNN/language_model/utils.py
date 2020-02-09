
import numpy as np


def softmax(x):
    xt= np.exp(x- np.max(x))
    return xt / np.sum(xt)


def save_model_parameters_theano(out_file, model):
    U, V, W= model.U.get_value(), model.V.get_value(), model.W.get_value()
    np.savez(out_file, U=U, V=V, W=W)


def load_model_parameters_theano(path, model):
    npz_file= np.load(path)
    U, V, W= npz_file['U'], npz_file['V'], npz_file['W']

    model.hidden_dim= U.shape[0]
    model.word_dim= U.shape[0]

    model.U.set_value(U)
    model.V.set_value(V)
    model.W.set_value(W)
