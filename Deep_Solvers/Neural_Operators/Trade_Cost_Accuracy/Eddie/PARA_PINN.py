import sys
import numpy as np
from mynn import *
from mydata import *

from Adam import Adam
import operator
from functools import reduce
from functools import partial
import matplotlib.pyplot as plt
from timeit import default_timer
import numpy as np
from torch.autograd import Variable


torch.manual_seed(0)
np.random.seed(0)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

M = 312
N = 100

ntrain = M//2
acc = 0.99

xgrid = np.linspace(0,1,N+1)
dx    = xgrid[1] - xgrid[0]


inputs = np.load("../../../../data/Helmholtz_inputs.npy")   
outputs = np.load("../../../../data/Helmholtz_outputs.npy")   

compute_input_PCA = False

train_inputs = np.reshape(inputs[:,:,:M//2], (-1, M//2))
Ui,Si,Vi = np.linalg.svd(train_inputs)
en_f= 1 - np.cumsum(Si)/np.sum(Si)
r_f = np.argwhere(en_f<(1-acc))[0,0]
r_f = 101 #r_f = min(r_f, 512)
#print("Energy is ", en_f[r_f - 1])
Uf = Ui[:,:r_f]
f_hat = np.matmul(Uf.T,train_inputs)

if compute_input_PCA:
    x_train_part = f_hat.T.astype(np.float32)
else:
    r_f = train_inputs.shape[0]
    x_train_part = train_inputs.T.astype(np.float32)


del Ui, Vi, Uf, f_hat
del train_inputs,inputs

Y, X = np.meshgrid(xgrid, xgrid)
# test
i = 20
j = 40
assert(X[i, j] == i*dx and Y[i, j] == j*dx)

X_upper = np.reshape(X, -1)
Y_upper = np.reshape(Y, -1)
N_upper = len(X_upper)
x_train = np.zeros((M//2 * N_upper, r_f + 2), dtype = np.float32)
y_train = np.zeros(M//2 * N_upper, dtype = np.float32)



for i in range(M//2):
    d_range = range(i*N_upper, (i + 1)*N_upper)
    x_train[d_range , 0:r_f]   = x_train_part[i, :]
    x_train[d_range , r_f]     = X_upper
    x_train[d_range , r_f + 1] = Y_upper 
    y_train[d_range] = np.reshape(outputs[:, :, i], -1)


print("Input dim : ", r_f+2, " output dim : ", 1)


x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train).unsqueeze(-1)

# x_normalizer = UnitGaussianNormalizer(x_train)
# x_normalizer.encode_(x_train)
# y_normalizer = UnitGaussianNormalizer(y_train)
# y_normalizer.encode_(y_train)

batch_size = 8192

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)


def de(self,data_domain,w=1e3):
    """ The pytorch autograd version of calculating residual """
    u = self(data_domain)

    du = torch.autograd.grad(
        u, data_domain, 
        grad_outputs=torch.ones_like(u),
        retain_graph=True,
        create_graph=True
    )[0]

    ddu_x = torch.autograd.grad(
        du[:,-2],data_domain, 
        grad_outputs=torch.ones_like(du[:,-2]),
        retain_graph=True,
        create_graph=True
        )[0]
    
    ddu_y = torch.autograd.grad(
        du[:,-1],data_domain, 
        grad_outputs=torch.ones_like(du[:,-1]),
        retain_graph=True,
        create_graph=True
        )[0]
    
    f = ddu_x[:,-2].reshape(-1,1)+ddu_y[:,-1].reshape(-1,1) + ((w/data_domain[:,:-2])**2)*u.reshape(-1,1)
    return f

def bc_b(self,data_bc):
#    data_bc = torch.cat ([data_bc[:,:2],torch.zeros((data_bc.shape[0],1),requires_grad=True) ],axis =1)
    u = self(data_bc)
    du = torch.autograd.grad(
        u, data_bc, 
        grad_outputs=torch.ones_like(u),
        retain_graph=True,
        create_graph=True
    )[0]
    return du[:,1].reshape(-1,1)

def bc_l(self,data_bc):
    #data_bc = torch.cat ([data_bc[:,0].reshape(-1,1),torch.zeros((data_bc.shape[0],1),requires_grad=True), data_bc[:,2].reshape(-1,1)],axis =1)
    u = self(data_bc)
    
    du = torch.autograd.grad(
        u, data_bc, 
        grad_outputs=torch.ones_like(u),
        retain_graph=True,
        create_graph=True
    )[0]
    return du[:,0].reshape(-1,1)

def bc_r(self,data_bc):
    #data_bc = torch.cat ([data_bc[:,0].reshape(-1,1),torch.ones((data_bc.shape[0],1),requires_grad=True), data_bc[:,2].reshape(-1,1)],axis =1)

    u = self(data_bc)
    
    du = torch.autograd.grad(
        u, data_bc, 
        grad_outputs=torch.ones_like(u),
        retain_graph=True,
        create_graph=True
    )[0]
    return du[:,0].reshape(-1,1)


def bc_u(self,data_bc):
    #data_bc = torch.cat ([data_bc[:,:2],torch.ones((data_bc.shape[0],1),requires_grad=True) ],axis =1)

    u = self(data_bc)
    
    du = torch.autograd.grad(
        u, data_bc, 
        grad_outputs=torch.ones_like(u),
        retain_graph=True,
        create_graph=True
    )[0]

    f = torch.where((data_bc[:,0] > 0.35) & (data_bc[:,0] < 0.65), 1, 0)
    return du[:,1].reshape(-1,1) - f.reshape(-1,1)

FNN.de = de
FNN.bc_b = bc_b
FNN.bc_l = bc_l
FNN.bc_r = bc_r
FNN.bc_u = bc_u


learning_rate = 0.001

epochs = 50

step_size = 100
gamma = 0.5
layers = 4

model = FNN(r_f + 2, 1, layers, 16) 
print(count_params(model))
model.to(device)

optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

myloss = torch.nn.MSELoss(reduction='sum')

t0 = default_timer()
for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_l2 = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        batch_size_ = x.shape[0]
        optimizer.zero_grad()
        out = model(x)
        #out = y_normalizer.decode(out)
        #y = y_normalizer.decode(y)
        pde_f = model.de(Variable(x,requires_grad=True)).mean(axis = 1)
        loss_pde = myloss(pde_f, torch.zeros_like(pde_f))
    
        Loss = myloss(out , y)
        loss = Loss +  loss_pde

        loss.backward()

        optimizer.step()
        train_l2 += loss.item()

    scheduler.step()

    train_l2/= ntrain

    t2 = default_timer()
    print("Epoch : ", ep, " Epoch time : ", t2-t1, " Train L2 Loss : ", train_l2)

torch.save(model, "./models/ParaNet_PINN_"+str(16)+"Nd_"+str(ntrain)+".model")
