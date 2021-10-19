

def test(**kwargs):
    if 'test' in kwargs :
        print(kwargs['test'])


test(test1 = 10)