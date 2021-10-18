
from re import M
from blocks import  * 


hyperparameters = {
    'in_ch': 3, 
    'out_ch': 10,
    'stride': 1,
    'pad': 'same',
    'dilation': 0,
    'act_fn': ReLU(),
    'kernel_shape': (3,3),
    'weights_init_method': 'he_uniform'
}

x = np.random.rand(10, 20, 20, 3)

layer = Conv2D(hyperparameters)
optimizer = SGD() 


y = layer.forward(x)
dLdX = layer.backward(np.ones_like(y))
optimized = optimizer(layer.trainable_parameters['w'], layer.gradients['w'], 'w')

for i in [x, y, dLdX, optimized]:
    print(i.shape)

print("done")