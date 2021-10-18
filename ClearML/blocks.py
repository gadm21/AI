

from copy import deepcopy 
from utils import * 


class SGD :

    def __init__(self, lr = 0.0001):
        self.cache = {} 
        self.cur_step = 0 
        self.momentum = 0
        self.hyperparameters = {} 
        self.lr = lr
    
    def __call__(self, param, param_grad, param_name, cur_loss = None):
        return self.update(param, param_grad, param_name, cur_loss) 
    
    @property 
    def hyperparameters(self):
        return {
            'cache': self.cache, 
            'momentum': self.momentum,
            'lr': self.lr 
        }


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
        pass 

    def __call__(self, x):
        if x.ndim == 1 :
            x = x.reshape(1, -1) 
        return self.fn(x) 
    
    def fn(self, x):
        return np.clip(x, 0, np.inf)

    def grad(self, x):
        return (x > 0).astype(int) 
        



class Conv2D :

    def __init__(self, hyperparameters):
        self.x = [] 
        
        self.hyperparameters = hyperparameters
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
        self.derived_variables = {'z': [], 'out_rows': [], 'out_cols': []}


    def set_params(self, summary_dict):
        pass 

    @property 
    def hyperparameters(self):
        return self.hyperparameters


    def forward(self, x, mode = 'train'):

        n_samples, rows, cols, in_ch = x.shape 
        w, b = self.trainable_parameters['w'], self.trainable_parameters['b']

        z = convolve(x, w, self.stride, self.pad, self.dilation) + b 
        y = self.act_fn(z) 

        if mode == 'train':
            self.x.append(x) 
            self.derived_variables['z'].append(z)
            self.derived_variables['out_rows'].append(z.shape[1])
            self.derived_variables['out_cols'].append(z.shape[2])
        
        return y 


    def backward(self, dLdY, mode = 'train'):
        
        if not isinstance(dLdY, list):
            dLdY = [dLdY]

        w, b = self.trainable_parameters['w'], self.trainable_parameters['b']
        xs, zs = self.x, self.derived_variables['z']
        (f_rows, f_cols), s, p, d = \
            self.kernel_shape, self.stride, self.pad, self.dilation

        dLdXs = []
        for x, z, dy in zip(xs, zs, dLdY):
            n_samples, out_rows, out_cols, out_ch = dy.shape 
            x_padded, (br1, br2, bc1, bc2) = \
                pad2D(x, p, self.kernel_shape, s, d) 
            
            dz = dy * self.act_fn.grad(z)
            dx, dw, db = \
                np.zeros_like(x_padded), np.zeros_like(w), np.zeros_like(b) 
            for m in range(n_samples):
                for row in range(out_rows):
                    for col in range(out_cols):
                        for ch in range(out_ch):

                            row0, row1 = row*s, (row*s) + f_rows*(d+1) -d 
                            col0, col1 = col*s, (col*s) + f_cols*(d+1) -d 

                            kernel = dz[m, row, col, ch]
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

    



