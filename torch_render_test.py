import numpy as np
import math
import torch
import cv2
from torch_render import Setup_Config_Lightfield
import torch_render
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--work_space",default="test_rendering/")
    args = parser.parse_args()

    os.makedirs(args.work_space,exist_ok=True)

    device = torch.device("cpu:0")

    standard_rendering_parameters = {}
    standard_rendering_parameters["config_dir"] = "wallet_of_torch_renderer/light_field/"
    standard_rendering_parameters["device"] = device


    setup = Setup_Config_Lightfield(standard_rendering_parameters)
    cam_pos_list_torch = [setup.get_cam_pos_torch(device)]

    ####################################
    ### load test data               ###
    ####################################
    data = np.fromfile("wallet_of_torch_renderer/render_test_params.bin",np.float32).reshape([-1,11])[:50]
    test_params = data[:,3:-1]#np.fromfile(args.work_space+"test_params.bin",np.float32).reshape([-1,11])
    test_positions = data[:,:3]#np.fromfile(args.work_space+"test_positions.bin",np.float32).reshape([-1,3])
    test_rottheta = data[:,[-1]]#np.fromfile(args.work_space+"test_rottheta.bin",np.float32).reshape([-1,1])
    
    ####################################
    ### rendering here               ###
    ####################################

    tmp_params = test_params
    tmp_positions = test_positions
    tmp_rottheta = test_rottheta
    rotate_theta_zero = torch.zeros(test_params.shape[0],1,dtype=torch.float32,device=device)#TODO this should be avoided!
    
    input_params = torch.from_numpy(tmp_params).to(device)
    input_positions = torch.from_numpy(tmp_positions).to(device)
    input_rotatetheta = torch.from_numpy(tmp_rottheta).to(device)

    used_rottheta = rotate_theta_zero
    # used_rottheta = input_rotatetheta
    ground_truth_lumitexels_direct,_ = torch_render.draw_rendering_net(setup,input_params,input_positions,used_rottheta,"ground_truth_renderer_direct")#[batch,lightnum,1]

    mask_state = torch.ones(setup.get_mask_num())
    mask_state[:120] = 0.6
    mask_state[:32] = 0.1
    mask_state[:13] = 0.0

    visibility_mask,_ = torch_render.get_mask_state_wrt_light(setup,input_positions,mask_state)

    test_node = ground_truth_lumitexels_direct
    test_node = test_node*5e5
    result = test_node.cpu().numpy()
    
    imgs = torch_render.visualize_lumi(result,setup)
    imgs_visinility_mask = torch_render.visualize_lumi(visibility_mask,setup)

    for idx,a_img in enumerate(imgs):
        cv2.imwrite(args.work_space+"{}.png".format(idx),a_img)
        cv2.imwrite(args.work_space+"{}_mask.png".format(idx),imgs_visinility_mask[idx]*255.0)
    
    print("at the end")