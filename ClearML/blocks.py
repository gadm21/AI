

from copy import deepcopy 
from utils import * 
from collections import defaultdict 

class SGD :

    def __init__(self, lr = 0.0001):
        self.cache = {} 
        self.cur_step = 0 
        self.momentum = 0
        self.hyperparameters = {} 
        self.lr = lr
    
    def __call__(self, param, param_grad, param_name, cur_loss = None):
        return self.update(param, param_grad, param_name, cur_loss) 
    

    def update(self, param, param_grad, param_name, cur_loss):
        c = self.cache 
        h = self.hyperparameters 

        if param_name not in c :
            self.cache[param_name] = np.zeros_like(param_grad) 
        
        t = np.inf 
        if norm(param_grad) > t :
            param_grad = (param_grad * t) / norm(param_grad) 
        
        update = self.momentum * c[param_name] + self.lr * param_grad
        self.cache = update 
        return param - update 


    def step(self):
        self.cur_step += 1
    
    def reset_step(self):
        self.cur_step = 0 
    
    def copy(self):
        return deepcopy(self)


class ReLU :

    def __init__(self):
        self.cache = defaultdict(list) 


    def __call__(self, x, mode = 'train'):
        return self.forward(x, mode)


    def forward(self, x, mode = 'train'):
        if mode == 'train':
            self.cache['x'].append(x) 

        if x.ndim == 1 :
            x = x.reshape(1, -1) 
        return np.clip(x, 0, np.inf) 


    def grad(self, x):
        return (x > 0).astype(int) 


    def backward(self, dLdYs, mode = 'train'):
        
        if not isinstance(dLdYs, list):
            dLdYs = [dLdYs]

        dLdXs = [dRdX*dLdY for dRdX, dLdY in zip(self.cache['x'], dLdYs)]

        return dLdXs[0] if len(dLdXs) == 1 else dLdXs  


class Flatten :

    def __init__(self, keep_dims = 'first'):

        self.cache = defaultdict(list)
        self.keep_dims = keep_dims
    
    def __call__(self, x, mode = 'train'):
        return self.forward(x, mode) 
    
    def forward(self, x, mode = 'train'):

        self.cache['input_shapes'].append(x.shape) 
        
        if self.keep_dims == 'first' : flattened_x = x.reshape(x.shape[0], -1) 
        elif self.keep_dims == 'last': flattened_x = x.reshape(x.shape[-1], -1) 
        else: raise ValueError('keeep_dims can only be \'first\' or \'last\'')

        return flattened_x 
    
    def backward(self, dLdY, mode = 'train'):

        if not isinstance(dLdY, list):
            dLdY = [dLdY]
        
        out = [dy.reshape(sh) for sh, dy in zip(self.cache['input_shapes'], dLdY)]
        return out[0] if len(out) == 1 else out 



class FullyConnected :

    def __init__(self, hyperparameters):

        self.hyperparameters = hyperparameters 
        in_dim, out_dim = hyperparameters['in_dim'], hyperparameters['out_dim']

        w, b = \
            init_weights(hyperparameters['weights_init_method'],(in_dim, out_dim)), np.zeros((1, out_dim))
        
        self.cache = defaultdict(list)
        self.trainable_parameters = {'w': w, 'b': b}
        self.gradients = {'w': np.zeros_like(w), 'b': np.zeros_like(b)}
        self.trainable = True 
    

    def __call__(self, x, mode = 'train'):
        return self.forward(x, mode) 

    def forward(self, x, mode = 'train'):

        y = np.dot(x, self.trainable_parameters['w']) + self.trainable_parameters['b']

        if mode == 'train':
            self.cache['x'].append(x) 
        
        return y 
    

    def backward(self, dLdYs, mode = 'train'):

        if not isinstance(dLdYs, list):
            dLdYs = [dLdYs] 
        
        dLdXs = [] 
        for x, dLdY in zip(self.cache['x'], dLdYs):
            dLdX = self.trainable_parameters['w'] * dLdY 
            dLdXs.append(dLdX) 

            dw, db = x.T * dLdY, dLdY.sum(axis = 0, keepdims = True) 

            if mode == 'train':
                self.gradients['w'] += dw 
                self.gradients['b'] += db 
        
        return dLdXs if len(dLdXs) == 1 else dLdXs 
            
        



class Conv2D :

    def __init__(self, hyperparameters):
        self.xs = [] 
        
        self.hyperparameters = hyperparameters
        for k, v in hyperparameters.items() :
            if isinstance(v, np.ndarray): print(k, v.shape) 
            else: print(k, v)
        self.trainable = True

        self.unfold_hyperparameters()
        self.init_trainable_parameters()


    def init_trainable_parameters(self):
        f_r, f_c = self.kernel_shape
        w = init_weights(
            method = self.weights_init_method,
            shape = (f_r, f_c, self.in_ch, self.out_ch)
        )
        b = np.zeros((1, 1, 1, self.out_ch))

        self.trainable_parameters = {'w': w, 'b': b}
        self.gradients = {'w': np.zeros_like(w), 'b': np.zeros_like(b)}

        print("initialization")
        
        print("\nparameters:")
        for k, v in self.trainable_parameters.items() :
            print(k, v.shape) 
        
        print("\ngradients:")
        for k, v in self.gradients.items() :
            print(k, v.shape)
        


    def set_params(self, summary_dict):
        pass 

    def __call__(self, x, mode = 'train'):
        return self.forward(x, mode) 

    def forward(self, x, mode = 'train'):

        w, b = self.trainable_parameters['w'], self.trainable_parameters['b']

        z = convolve(x, w, self.stride, self.pad, self.dilation) + b 
        y = self.act_fn(z) 
        
        print("\noutputs")
        print('z:', z.shape) 
        print('y:', y.shape)

        if mode == 'train':
            self.xs.append(x) 
        
        return y 


    def backward(self, dLdY, mode = 'train'):
        
        if not isinstance(dLdY, list):
            dLdY = [dLdY]

        xs, w, b = self.xs, self.trainable_parameters['w'], self.trainable_parameters['b']

        (f_rows, f_cols), s, p, d = \
            self.kernel_shape, self.stride, self.pad, self.dilation

        dLdXs = []
        for x, dy in zip(xs, dLdY):
            n_samples, out_rows, out_cols, out_ch = dy.shape 
            x_padded, (br1, br2, bc1, bc2) = \
                pad2D(x, p, self.kernel_shape, s, d) 
            

            print('\nbackward variables:')
            print("padded x:", x_padded.shape) 


            dx, dw, db = \
                np.zeros_like(x_padded), np.zeros_like(w), np.zeros_like(b) 
            for m in range(n_samples):
                for row in range(out_rows):
                    for col in range(out_cols):
                        for ch in range(out_ch):

                            row0, row1 = row*s, (row*s) + f_rows*(d+1) -d 
                            col0, col1 = col*s, (col*s) + f_cols*(d+1) -d 

                            kernel = dy[m, row, col, ch]
                            db[:, :, :, ch] += kernel 

                            wc = w[:, :, :, ch]
                            window = x_padded[m, row0:row1:(d+1), col0:col1:(d+1), :]

                            dw[:, :, :, ch] += window * kernel
                            dx[m, row0:row1:(d+1), col0:col1:(d+1), :] += \
                                wc * kernel 
            
            if mode == 'train':
                self.gradients['w'] += dw 
                self.gradients['b'] += db 
            
            br2 = None if br2 == 0 else -br2 
            bc2 = None if bc2 == 0 else -bc2 
            dLdXs.append(dx[:, br1:br2, bc1:bc2, :])
        
        return dLdXs[0] if len(dLdXs) == 1 else dLdXs
    

    def update(self, cur_loss = None):
        pass 


    def summary(self):
        pass 

    def unfold_hyperparameters(self):
        self.in_ch, self.out_ch = \
            self.hyperparameters['in_ch'], self.hyperparameters['out_ch']
        self.stride, self.pad, self.dilation = \
            self.hyperparameters['stride'], self.hyperparameters['pad'], self.hyperparameters['dilation']
        
        self.act_fn, self.kernel_shape = \
            self.hyperparameters['act_fn'], self.hyperparameters['kernel_shape']
        
        self.weights_init_method = self.hyperparameters['weights_init_method']

    



