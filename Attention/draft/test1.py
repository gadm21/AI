import torch









def run():

    x = torch.arange(3)[None, :]
    z = x.expand(2, -1).contiguous()
    x[0, 0] = 34 
    xx = x.numpy()
    zz = z.numpy()

    print("shape:",xx.shape, " ", zz.shape)
    print(xx)
    print()
    print(zz)

if __name__=='__main__':
    run()