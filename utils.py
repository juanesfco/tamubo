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
    #elif torch.mps.is_available():
    #    if x.dtype != torch.float32:
    #        x = x.to(torch.float32)
    #    device = torch.device('mps')
    #    x = x.to(device)
    #    return(x)
    else:
        return(x)
    
def torch_squareform(x):
    """
    If x is a 1D tensor, returns the corresponding square 2D tensor.
    If x is a 2D square tensor, returns the condensed 1D form.
    """
    if x.dim() == 1:
        # Determine size of output matrix
        n = int((1 + (1 + 8 * x.size(0))**0.5) / 2)
        if n * (n - 1) // 2 != x.size(0):
            raise ValueError("Invalid size for condensed vector.")
        out = torch.zeros((n, n), dtype=x.dtype, device=x.device)
        idx = torch.triu_indices(n, n, offset=1)
        out[idx[0], idx[1]] = x
        out[idx[1], idx[0]] = x
        return out
    elif x.dim() == 2 and x.size(0) == x.size(1):
        # Extract upper triangle without diagonal
        idx = torch.triu_indices(x.size(0), x.size(0), offset=1)
        return x[idx[0], idx[1]]
    else:
        raise ValueError("Input must be a 1D or square 2D tensor.")