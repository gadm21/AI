


import numpy as np 
from numpy.linalg import norm









def init_weights(method, shape):

    def calc_fan(shape):
        if len(shape) == 2 :
            fan_in, fan_out = shape 
        elif len(shape) in [3, 4] :
            in_ch, out_ch = shape[-2:] 
            kernel_size = np.prod(shape[:-2])
            fan_in, fan_out = in_ch * kernel_size, out_ch * kernel_size
        else: raise ValueError('unrecognized weights dimensions')

        return fan_in, fan_out

    def truncated_normal(mean, std, out_shape) :
        samples = np.random.normal(loc = mean, scale = std, size = out_shape)
        reject = np.logical_or(samples >= mean + (2*std), samples <= mean - (2*std))
        while any(reject.flatten()):
            resamples = np.random.normal(loc = mean, scale = std, size = reject.sum())
            samples[reject] = resamples 
            reject = np.logical_or(samples >= mean + (2*std), samples <= mean - (2*std))
        return samples 


    def he_uniform(shape):
        fan_in, fan_out = calc_fan(shape) 
        b = np.sqrt(6 / fan_in) 
        return np.random.uniform(-b, b, size = shape) 
    
    def he_normal(shape):
        fan_in, fan_out = calc_fan(shape) 
        std = np.sqrt(2 / fan_in) 
        return truncated_normal(0, std, shape) 

    if method == 'he_uniform':
        return he_uniform(shape) 
    elif method == 'he_normal':
        return he_normal(shape) 
    else: raise ValueError('unrecognized weights initialization method')






def calc_pad_dims_2D(X_shape, out_dim, kernel_shape, stride, dilation=0):
    """
    Compute the padding necessary to ensure that convolving `X` with a 2D kernel
    of shape `kernel_shape` and stride `stride` produces outputs with dimension
    `out_dim`.

    Parameters
    ----------
    X_shape : tuple of `(n_ex, in_rows, in_cols, in_ch)`
        Dimensions of the input volume. Padding is applied to `in_rows` and
        `in_cols`.
    out_dim : tuple of `(out_rows, out_cols)`
        The desired dimension of an output example after applying the
        convolution.
    kernel_shape : 2-tuple
        The dimension of the 2D convolution kernel.
    stride : int
        The stride for the convolution kernel.
    dilation : int
        Number of pixels inserted between kernel elements. Default is 0.

    Returns
    -------
    padding_dims : 4-tuple
        Padding dims for `X`. Organized as (left, right, up, down)
    """
    if not isinstance(X_shape, tuple):
        raise ValueError("`X_shape` must be of type tuple")

    if not isinstance(out_dim, tuple):
        raise ValueError("`out_dim` must be of type tuple")

    if not isinstance(kernel_shape, tuple):
        raise ValueError("`kernel_shape` must be of type tuple")

    if not isinstance(stride, int):
        raise ValueError("`stride` must be of type int")

    d = dilation
    fr, fc = kernel_shape
    out_rows, out_cols = out_dim
    n_ex, in_rows, in_cols, in_ch = X_shape

    # update effective filter shape based on dilation factor
    _fr, _fc = fr * (d + 1) - d, fc * (d + 1) - d

    pr = int((stride * (out_rows - 1) + _fr - in_rows) / 2)
    pc = int((stride * (out_cols - 1) + _fc - in_cols) / 2)

    out_rows1 = int(1 + (in_rows + 2 * pr - _fr) / stride)
    out_cols1 = int(1 + (in_cols + 2 * pc - _fc) / stride)

    # add asymmetric padding pixels to right / bottom
    pr1, pr2 = pr, pr
    if out_rows1 == out_rows - 1:
        pr1, pr2 = pr, pr + 1
    elif out_rows1 != out_rows:
        raise AssertionError

    pc1, pc2 = pc, pc
    if out_cols1 == out_cols - 1:
        pc1, pc2 = pc, pc + 1
    elif out_cols1 != out_cols:
        raise AssertionError

    if any(np.array([pr1, pr2, pc1, pc2]) < 0):
        raise ValueError(
            "Padding cannot be less than 0. Got: {}".format((pr1, pr2, pc1, pc2))
        )
    return (pr1, pr2, pc1, pc2)

def pad2D(X, p, kernel_shape = None, stride = None, dilation = 0):
    
    if isinstance(p, int):
        p = (p, p, p, p)

    if isinstance(p, tuple):
        if len(p) == 2:
            p = (p[0], p[0], p[1], p[1])

        X_pad = np.pad(
            X,
            pad_width=((0, 0), (p[0], p[1]), (p[2], p[3]), (0, 0)),
            mode="constant",
            constant_values=0,
        )

    # compute the correct padding dims for a 'same' convolution
    if p == "same" and kernel_shape and stride is not None:
        p = calc_pad_dims_2D(
            X.shape, X.shape[1:3], kernel_shape, stride, dilation=dilation
        )
        X_pad, p = pad2D(X, p)
    return X_pad, p

def convolve(x, w, s, p, d):
    x_padded, boundaries = pad2D(x, p, w.shape[:2], s, d) 
    br1, br2, bc1, bc2 = boundaries
    f_r, f_c, in_ch, out_ch = w.shape 
    n_samples, in_rows, in_cols, in_ch = x.shape 
    f_r, f_c = f_r * (d+1)-d, f_c * (d+1)-d

    out_rows = int((in_rows+br1+br2-f_r) / s+1) 
    out_cols = int((in_cols+bc1+bc2-f_c) / s+1) 

    z = np.zeros((n_samples, out_rows, out_cols, out_ch)) 
    for m in range(n_samples):
        for c in range(out_ch):
            for row in range(out_rows):
                for col in range(out_cols):
                    row0, row1 = row*s, (row*s)+f_r 
                    col0, col1 = col*s, (col*s)+f_c 

                    window = x_padded[m, row0:row1:(d+1), col0:col1:(d+1), :]
                    z[m, row, col, c] = np.sum(window*w[:, :, :, c])
    return z 


