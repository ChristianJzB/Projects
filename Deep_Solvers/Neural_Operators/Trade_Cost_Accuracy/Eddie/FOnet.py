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
    ntest = M-M//2
    s = N+1
    acc = 0.99

    xgrid = np.linspace(0,1,N+1)
    dx    = xgrid[1] - xgrid[0]



    inputs = np.load("./data/Helmholtz_inputs.npy")   
    outputs = np.load("./data/Helmholtz_outputs.npy")  

    inputs = inputs.transpose(2, 0, 1)
    outputs = outputs.transpose(2, 0, 1)
 

    x_train = torch.from_numpy(np.reshape(inputs[:M//2, :, :], -1).astype(np.float32))
    y_train = torch.from_numpy(np.reshape(outputs[:M//2, :, :], -1).astype(np.float32))

    x_test = torch.from_numpy(np.reshape(inputs[M//2:M, :, :], -1).astype(np.float32))
    y_test = torch.from_numpy(np.reshape(outputs[M//2:M, :, :], -1).astype(np.float32))

    x_normalizer = UnitGaussianNormalizer(x_train)
    x_train = x_normalizer.encode(x_train)
    x_test = x_normalizer.encode(x_test)

    y_normalizer = UnitGaussianNormalizer(y_train)
    y_train = y_normalizer.encode(y_train)


    x_train = x_train.reshape(ntrain,s,s,1)
    x_test = x_test.reshape(ntest,s,s,1)

    # todo do we need this
    y_train = y_train.reshape(ntrain,s,s,1)
    y_test = y_test.reshape(ntest,s,s,1)



    batch_size = 1024

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)


    learning_rate = 0.001

    epochs = 500

    step_size = 100
    gamma = 0.5

    modes = [2,4,8,16,32]
    N_neurons = [16,64,128,256]

    for nneurons, mode in zip(N_neurons,modes):

        model = FNO2d(mode, mode, nneurons)
        print(count_params(model))
        model.to(device)

        optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

        myloss = torch.nn.MSELoss(reduction='sum')
        #####
        #y_normalizer.cuda()
        #y_normalizer.cpu()
        t0 = default_timer()
        for ep in range(epochs):
            model.train()
            t1 = default_timer()
            train_l2 = 0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)

                batch_size_ = x.shape[0]
                optimizer.zero_grad()
                out = model(x).reshape(batch_size_, s, s)
                out = y_normalizer.decode(out)
                y = y_normalizer.decode(y)

                loss = myloss(out.view(batch_size_,-1), y.view(batch_size_,-1))
                loss.backward()

                optimizer.step()
                train_l2 += loss.item()

            scheduler.step()

            train_l2/= ntrain

            t2 = default_timer()
            print("Epoch : ", ep, " Epoch time : ", t2-t1, " Rel. Train L2 Loss : ", train_l2)

        print("Total time is :", default_timer() - t0, "Total epoch is ", epochs)
        torch.save(model, "./models/FNO_"+str(nneurons)+"Nd_"+str(ntrain)+".model")
