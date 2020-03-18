import numpy as np
import torch
# from multiview_renderer import Multiview_Renderer
from multiview_renderer_naive import Multiview_Renderer
import time

if __name__ == "__main__":
    test_configs = {
        # "available_devices":[torch.device("cuda:0")],
        "available_devices":[torch.device("cuda:0"),torch.device("cuda:1"),torch.device("cuda:2")],
        "torch_render_path":"./",
        "sample_view_num":24
    }

    renderer = Multiview_Renderer(test_configs)

    batch_size = 50
    test_params = np.random.rand(batch_size,7)
    test_positions = np.random.rand(batch_size,3)

    
    tmp_params = torch.from_numpy(test_params.astype(np.float32)).to("cuda:0")
    tmp_positions = torch.from_numpy(test_positions.astype(np.float32)).to("cuda:0")
    
    for itr in range(200):
        if itr % 10 == 0:
            print("-------------------itr:{}".format(itr))
        if itr == 10:
            start = time.time()
        renderer(tmp_params,tmp_positions)
    end = time.time()
    print(end - start)