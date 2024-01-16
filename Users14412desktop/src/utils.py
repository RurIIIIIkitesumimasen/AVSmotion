import torch

def make_shape(model):
    W = model.state_dict()['dconvSynaps.W']
    q = model.state_dict()['dconvSynaps.q']
    W = W.permute(2,0,1)
    q = q.permute(2,0,1)
    neuron_model = torch.zeros_like(W)
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            for k in range(W.shape[2]):
                if 0 < W[i,j,k] < q[i,j,k]:
                    neuron_model[i,j,k] = 0
                elif W[i,j,k] < 0 < q[i,j,k]:
                    neuron_model[i,j,k] = 0
                elif W[i,j,k] < q[i,j,k] < 0:
                    neuron_model[i,j,k] = 2
                elif 0 < q[i,j,k] < W[i,j,k]:
                    neuron_model[i,j,k] = 1
                elif q[i,j,k] < 0 < W[i,j,k]:
                    neuron_model[i,j,k] = 3
                elif q[i,j,k] < W[i,j,k] < 0:
                    neuron_model[i,j,k] = 3
    #print(neuron_model)

    #0:cons=0,1:direct,2:inverse,3:cons=1
    #label placing
    m = 0
    M = 3
    w = (neuron_model-m)/(M-m)
    W1 = w.reshape(-1,1,6,3)
    return W1