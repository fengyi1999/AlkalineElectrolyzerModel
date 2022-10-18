import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import pickle

def get_data():
    '''
    :return: tensor_input, tensor_expe
    '''
    file_name = '../data/kappa-T-wt.csv'
    data = pd.read_csv(file_name)
    T = list(float(i) for i in data.columns)[1:]   # C
    wt = list(data['NaN'])
    tensor_expe = torch.tensor(pd.array(data.iloc[:,1:],dtype=float).T,device=torch.device('cuda'),).reshape(len(T),len(wt),1)
    tensor_input = torch.zeros(len(T),len(wt),2, device=torch.device('cuda'),dtype=float)
    for i in range(len(T)):
        tensor_input[i,:,0] = T[i]
        for j in range(len(wt)):
            tensor_input[:,j,1] = wt[j]
    return tensor_input.reshape(len(T)*len(wt),2), tensor_expe.reshape(len(T)*len(wt),1), len(T)*len(wt)

def straight():
    data = get_data()[0]
    labels = get_data()[1]
    number = get_data()[2]
    weights0 = torch.randn((2,number), dtype=float, device=torch.device('cuda') ,requires_grad=True)
    biases0 = torch.randn(number, dtype=float, device=torch.device('cuda'), requires_grad=True)
    weights1 = torch.randn((number, 1), dtype=float, device=torch.device('cuda'), requires_grad=True)
    biases1 = torch.randn(1, dtype=float, device=torch.device('cuda'), requires_grad=True)

    learning_rate = 0.0000000001
    losses = []

    for i in range(1000000000000000):

        hidden = data.mm(weights0)+biases0
        hidden = torch.relu(hidden)
        predictions = hidden.mm(weights1) + biases1

        loss = torch.mean((predictions-labels)**2/labels)
        losses.append(loss.data)

        if i%1000 ==0:
            print('loss', loss)

        loss.backward()

        weights0.data.add_(-learning_rate * weights0.grad.data)
        weights1.data.add_(-learning_rate * weights1.grad.data)
        biases0.data.add_(-learning_rate * biases0.grad.data)
        biases1.data.add_(-learning_rate * biases1.grad.data)

        weights0.grad.data.zero_()
        weights0.grad.data.zero_()
        biases0.grad.data.zero_()
        biases0.grad.data.zero_()

def simplenn(times):
    x = get_data()[0]
    y = get_data()[1]
    input_size = x.shape[1]
    hidden_size = 128
    output_size = 1
    batch_size = 27
    my_nn = nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.Sigmoid(),
        nn.Linear(hidden_size, output_size),
    ).cuda()
    cost = nn.MSELoss(reduction='mean')
    opt = torch.optim.Adam(my_nn.parameters(), lr = 1e-5)
    losses = []
    # my_nn.load_state_dict(torch.load('m.pkl'))
    my_nn.load('m.ckpt')
    # for i in range(int(times)):
    #     batch_loss = []
    #     for start in range(0,len(x), batch_size):
    #         end = start + batch_size if start+batch_size<len(x) else len(x)
    #         xx = torch.tensor(x[start:end], device=torch.device('cuda'), dtype=torch.float, requires_grad=True)
    #         yy = torch.tensor(y[start:end], device=torch.device('cuda'), dtype=torch.float, requires_grad=True)
    #         prediction = my_nn(xx)
    #         loss = cost(prediction, yy)
    #         opt.zero_grad()
    #         loss.backward(retain_graph=True)
    #         opt.step()
    #         batch_loss.append(loss.data.cpu().detach().numpy())
    #     if i%100 ==0:
    #         losses.append(np.mean(batch_loss))
    #         print(i, np.mean(batch_loss))
    # torch.save(my_nn, 'm.ckpt')
out = simplenn(1e3)
# out = open('m.pkl', 'rb')
# print(out)
class KappaPredict(nn.Module):
    def __init__(self):
        super(KappaPredict, self).__init__()
        x = get_data()[0]
        y = get_data()[1]
        input_size = x.shape[1]
        hidden_size = 128
        output_size = 1
        batch_size = 27

        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Sigmoid(),
            nn.Linear(hidden_size, output_size),
        )
    def forward(self,x):
        return self.layers(x)
    