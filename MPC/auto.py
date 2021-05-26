
from utils import *





## prediction horizen ##
P = 10
acc_offest = P
## init geuss ##
# two variables for every time step. In every optimization step, the algorithm decides P decisions
# in the future where only the dicision for the first step is used and then the optimization is 
# called again to predict the next P steps in the future and so on.
x0 = np.zeros(2*P) 




## input bounds ##
b1 = (-1.22, 1.22)  # steering bounds
b2 = (0.0, 1.0)   # acceleration bounds
bnds = tuple([b1]*P+[b2]*P)
# bnds = (b1,b1,b1,b1,b1,b1,b1,b1,b1,b1,b2,b2,b2,b2,b2,b2,b2,b2,b2,b2)


print(bnds)
