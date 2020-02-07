import numpy as np
import math
import torch
import sys
sys.path.append("../utils/")
from lumitexel_related import visualize_init,visualize_new
from timer import MeasureDuration
from torch_render import Setup_Config
import torch_render
from parser_related import get_bool_type
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    get_bool_type(parser)
    parser.add_argument("work_space",default="/home/cocoa_kang/no_where/")
    parser.add_argument("task_name",choices={"gen_result"})
    args = parser.parse_args()

    if args.task_name == "gen_result":
        device = torch.device("cuda:0")
        batch_size = 5

        standard_rendering_parameters = {}
        standard_rendering_parameters["config_dir"] = "wallet_of_torch_renderer/blackbox20_render_configs_1x1/"
        standard_rendering_parameters["device"] = device


        setup = Setup_Config(standard_rendering_parameters)
        cam_pos_list_torch = [setup.get_cam_pos_torch()]

        ####################################
        ### load test data               ###
        ####################################
        test_params = np.fromfile(args.work_space+"test_params.bin",np.float32).reshape([-1,11])
        test_positions = np.fromfile(args.work_space+"test_positions.bin",np.float32).reshape([-1,3])
        test_rottheta = np.fromfile(args.work_space+"test_rottheta.bin",np.float32).reshape([-1,1])
        pf_output = open(args.work_space+"torch_result.bin","wb")

        ####################################
        ### rendering here               ###
        ####################################

        ptr = 0

        tmp_params = test_params[ptr:ptr+batch_size]
        tmp_positions = test_positions[ptr:ptr+batch_size]
        tmp_rottheta = test_rottheta[ptr:ptr+batch_size]
        rotate_theta_zero = torch.zeros(batch_size,1,dtype=torch.float32,device=device)#TODO this should be avoided!
        with MeasureDuration() as m:
            # for i in range(5000):
            for i in range(1):
                # if tmp_params.shape[0] == 0:
                #     break
                # else:
                #     ptr = ptr + tmp_params.shape[0]
                #     pass
                ###process start
                    

                
                with torch.no_grad():
                    input_params = torch.from_numpy(tmp_params).to(device)
                    input_positions = torch.from_numpy(tmp_positions).to(device)
                    input_rotatetheta = torch.from_numpy(tmp_rottheta).to(device)
                    n_2d,theta,axay,pd3,ps3 = torch.split(input_params,[2,1,2,3,3],dim=1)
                    n_local = torch_render.back_hemi_octa_map(n_2d)
                    t_local,_ = torch_render.build_frame_f_z(n_local,theta,device,with_theta=True)

                    view_dir = cam_pos_list_torch[0] - input_positions #shape=[batch,3]
                    view_dir = torch.nn.functional.normalize(view_dir,dim=1)#shape=[batch,3]
                    
                    #build local frame
                    frame_t,frame_b = torch_render.build_frame_f_z(view_dir,None,device,with_theta=False)#[batch,3]
                    frame_n = view_dir

                    n_local_x,n_local_y,n_local_z = torch.split(n_local,[1,1,1],dim=1)#[batch,1],[batch,1],[batch,1]

                    normal = n_local_x*frame_t+n_local_y*frame_b+n_local_z*frame_n#[batch,3]
                    t_local_x,t_local_y,t_local_z = torch.split(t_local,[1,1,1],dim=1)#[batch,1],[batch,1],[batch,1]
                    tangent = t_local_x*frame_t+t_local_y*frame_b+t_local_z*frame_n#[batch,3]
                    binormal = torch.cross(normal,tangent)#[batch,3]

                    global_frame = [normal,tangent,binormal]
                    
                    used_rottheta = rotate_theta_zero
                    used_rottheta = input_rotatetheta
                    ground_truth_lumitexels_direct = torch_render.draw_rendering_net(setup,device,input_params,input_positions,used_rottheta,"ground_truth_renderer_direct")#[batch,lightnum,1]
                    test_node = ground_truth_lumitexels_direct

        result = test_node.cpu().numpy()
        # print(result)
    
        result.astype(np.float32).tofile(pf_output)

        pf_output.close()