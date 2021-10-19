
from re import M
from blocks import  * 


hyperparameters = {
    'in_ch': 3, 
    'out_ch': 10,
    'stride': 1,
    'pad': 0,
    'dilation': 0,
    'act_fn': ReLU(),
    'kernel_shape': (5, 5),
    'weights_init_method': 'he_uniform'
}

x = np.random.rand(10, 20, 20, 3)

conv1 = Conv2D(hyperparameters)
flat1 = Flatten()
full1 = FullyConnected({
    'in_dim': 2560,
    'out_dim': 10,
    'weights_init_method': 'he_uniform'
})
optimizer = SGD() 


y = conv1(x)
f = flat1(y) 
d = full1(f)
# optimized = optimizer(layer.trainable_parameters['w'], layer.gradients['w'], 'w')

print(y.shape)
print(f.shape) 
print(d.shape)

print("done")