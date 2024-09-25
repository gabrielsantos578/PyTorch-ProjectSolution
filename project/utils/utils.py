import pandas as pd
import numpy as np
import torch

def to_tensor(data):
    # Verifica se o dado é uma Series ou DataFrame e converte para tensor
    if isinstance(data, (pd.Series, pd.DataFrame)):
        return torch.tensor(data.values, dtype=torch.float32)
    elif isinstance(data, np.ndarray):  # Para arrays numpy
        return torch.tensor(data, dtype=torch.float32)
    else:
        raise ValueError("Tipo de dado não suportado para conversão.")
