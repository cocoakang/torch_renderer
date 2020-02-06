import torch
import numpy as np

data = np.random.rand(5,1)
print(data)
data = torch.from_numpy(data)
result = torch.where(torch.gt(data, 0.5),torch.ones(5,3,dtype=torch.float32),torch.zeros(5,3,dtype=torch.float32))
print(result.cpu().numpy())