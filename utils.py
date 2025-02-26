import torch

def numpyToTorch(x):
    if torch.is_tensor(x):
        x = tensorToGPU(x)
        return(x)
    else:
        x = torch.Tensor(x)
        x = tensorToGPU(x)
        return(x)
    
def tensorToGPU(x):
    if torch.cuda.is_available():
        device = torch.device('cuda')
        x = x.to(device)
        return(x)
    elif torch.mps.is_available():
        #device = torch.device('mps')
        #x = x.to(device)
        return(x)
    else:
        return(x)