import torch

def numpyToTorch(x):
    if torch.is_tensor(x):
        x = tensorToGPU(x)
        return(x)
    else:
        x = torch.tensor(x)
        x = tensorToGPU(x)
        return(x)
    
def tensorToGPU(x):
    if torch.cuda.is_available():
        if x.dtype != torch.float64:
            x = x.double()
        device = torch.device('cuda')
        x = x.to(device)
        return(x)
    elif torch.mps.is_available():
        if x.dtype != torch.float32:
            x = x.to(torch.float32)
        device = torch.device('mps')
        x = x.to(device)
        return(x)
    else:
        return(x)