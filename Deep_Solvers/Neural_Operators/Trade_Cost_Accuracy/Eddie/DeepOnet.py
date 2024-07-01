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

torch.manual_seed(0)
np.random.seed(0)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

M_  = [312*(2**i) for i in range(6)]
for M in M_:
    N = 100

    ntrain = M//2
    acc = 0.99

    xgrid = np.linspace(0,1,N+1)
    dx    = xgrid[1] - xgrid[0]



    inputs = np.load("./data/Helmholtz_inputs.npy")   
    outputs = np.load("./data/Helmholtz_outputs.npy")   

    train_inputs = np.reshape(inputs[:,:,:M//2], (-1, M//2))
    Ui,Si,Vi = np.linalg.svd(train_inputs)
    en_f= 1 - np.cumsum(Si)/np.sum(Si)
    r_f = np.argwhere(en_f<(1-acc))[0,0]
    r_f = min(r_f, 512)
    print("Energy is ", en_f[r_f - 1])
    Uf = Ui[:,:r_f]
    f_hat = np.matmul(Uf.T,train_inputs)
    x_train_part = f_hat.T.astype(np.float32)

    del train_inputs
    del inputs
    del Ui, Vi, Uf, f_hat

    Y, X = np.meshgrid(xgrid, xgrid)
    # test
    i = 20
    j = 40
    assert(X[i, j] == i*dx and Y[i, j] == j*dx)

    X_upper = full2upper(X)
    Y_upper = full2upper(Y)
    N_upper = len(X_upper)
    x_train = np.zeros((M//2 * N_upper, r_f + 2), dtype = np.float32)
    y_train = np.zeros(M//2 * N_upper, dtype = np.float32)



    for i in range(M//2):
        d_range = range(i*N_upper, (i + 1)*N_upper)
        x_train[d_range , 0:r_f]   = x_train_part[i, :]
        x_train[d_range , r_f]     = X_upper
        x_train[d_range , r_f + 1] = Y_upper 
        y_train[d_range] = full2upper(outputs[:, :, i])


    print("Input dim : ", r_f+2, " output dim : ", 1)


    x_train = torch.from_numpy(x_train)
    y_train = torch.from_numpy(y_train).unsqueeze(-1)

    x_normalizer = UnitGaussianNormalizer(x_train)
    x_normalizer.encode_(x_train)
    y_normalizer = UnitGaussianNormalizer(y_train)
    y_normalizer.encode_(y_train)

    batch_size = 1024

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)


    learning_rate = 0.001

    epochs = 2

    step_size = 100
    gamma = 0.5
    layers = 4

    N_neurons = [16,64,128,256]

    for nneurons in N_neurons:

        model = DeepONet(r_f, 2,  layers,  layers, nneurons) 
        print(count_params(model))
        model.to(device)

        optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

        myloss = torch.nn.MSELoss(reduction='sum')
        #####
        y_normalizer.cpu()
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
                out = y_normalizer.decode(out)
                y = y_normalizer.decode(y)

                loss = myloss(out , y)
                loss.backward()

                optimizer.step()
                train_l2 += loss.item()

            #torch.save(model, "DeepONetNet_"+str(N_neurons)+"Nd_"+str(ntrain)+".model")
            scheduler.step()

            train_l2/= ntrain

            t2 = default_timer()
            print("Epoch : ", ep, " Epoch time : ", t2-t1, " Train L2 Loss : ", train_l2)

        torch.save(model, "./models/DeepONetNet_"+str(nneurons)+"Nd_"+str(ntrain)+".model")