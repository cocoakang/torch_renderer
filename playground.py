import torch
import numpy as np

data = np.random.rand(5,3)

idxes = np.argmax(data,axis=1)
# print(data)
print(idxes)

visulaize_idxes = np.arange(20).reshape([-1,2])
print(visulaize_idxes)

selected_idxes = visulaize_idxes[idxes]
print(selected_idxes)