import numpy as np
import cv2
import os
import argparse
import open3d as o3d

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_root",default="../wallet_of_torch_renderer/light_field/")

    args = parser.parse_args()

    os.makedirs(args.save_root,exist_ok=True)

    ##############
    ###lights
    ##############
    led_vertical_num = 64
    led_horizontal_num = 64
    led_size = 10.0#mm
    led_plane_x = 600.0#mm

    light_pos_collector = []
    light_normal_collector = []
    visualize_idx_collector = []
    for which_row in range(led_vertical_num):
        for which_col in range(led_horizontal_num):
            cur_light_pos_x = led_plane_x
            cur_light_pos_y = -led_horizontal_num//2*led_size+led_size/2+led_size*which_col
            cur_light_pos_z = -led_vertical_num//2*led_size+led_size/2 + led_size*which_row

            cur_light_pos = np.array((cur_light_pos_x,cur_light_pos_y,cur_light_pos_z),np.float32)
            light_pos_collector.append(cur_light_pos)

            cur_light_normal = np.array((1.0,0.0,0.0))
            light_normal_collector.append(cur_light_normal)

            cur_visualize_idx = np.array((which_col,led_vertical_num-which_row-1),np.int32)
            visualize_idx_collector.append(cur_visualize_idx)
    
    light_poses = np.stack(light_pos_collector,axis=0)
    light_normals = np.stack(light_normal_collector,axis=0)
    light_all = np.stack((light_poses,light_normals),axis=0)
    light_all.astype(np.float32).tofile(args.save_root+"lights.bin")

    ##############
    ###light visualization
    ##############
    with open(args.save_root+"visualize_config_torch.bin","wb") as pf:
        img_size = np.array((led_vertical_num,led_horizontal_num),np.int32)#height,width
        visualize_map = np.stack(visualize_idx_collector,axis=0)#(lightnum,2) x,y
        img_size.tofile(pf)
        visualize_map.tofile(pf)

    ##############
    ###mask
    ##############
    mask_vertical_num = 32
    mask_horizontal_num = 32
    mask_size = 20.0#mm
    mask_plane_x = 580.0#mm

    mask_pos_collector = []
    visualize_idx_collector = []
    for which_row in range(mask_vertical_num):
        for which_col in range(mask_horizontal_num):
            cur_mask_pos_x = mask_plane_x
            cur_mask_pos_y = -mask_horizontal_num//2*mask_size+mask_size/2+mask_size*which_col
            cur_mask_pos_z = -mask_vertical_num//2*mask_size+mask_size/2 + mask_size*which_row

            cur_mask_pos = np.array((cur_mask_pos_x,cur_mask_pos_y,cur_mask_pos_z),np.float32)
            mask_pos_collector.append(cur_mask_pos)

            cur_visualize_idx = np.array((which_col,mask_vertical_num-which_row-1),np.int32)
            visualize_idx_collector.append(cur_visualize_idx)
    
    mask_poses = np.stack(mask_pos_collector,axis=0)
    mask_all = np.stack((mask_poses),axis=0)
    mask_all.astype(np.float32).tofile(args.save_root+"masks.bin")

    ##############
    ###lmask visualization
    ##############
    with open(args.save_root+"visualize_config_mask.bin","wb") as pf:
        img_size = np.array((mask_vertical_num,mask_horizontal_num),np.int32)#height,width
        visualize_map = np.stack(visualize_idx_collector,axis=0)#(lightnum,2) x,y
        img_size.tofile(pf)
        visualize_map.tofile(pf)

    ##############
    ###camera
    ##############
    max_led_pos_z = np.max(light_poses[:,2])
    cam_pos = np.array((led_plane_x-10.0,0.0,max_led_pos_z+30.0),np.float32)
    cam_pos.astype(np.float32).tofile(args.save_root+"cam_pos.bin")

    points_collector = np.concatenate(
        [
            light_poses,
            cam_pos.reshape((1,3)),
            mask_poses
        ],axis=0
    )
    normal_collector = np.concatenate(
        [
            light_normals,
            np.zeros((1,3),np.float32),
            np.zeros((mask_poses.shape[0],3),np.float32)
        ],axis=0
    )

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_collector)
    pcd.normals = o3d.utility.Vector3dVector(normal_collector)
    o3d.io.write_point_cloud(args.save_root+"setup.ply", pcd)

    ######################################################################
    ###test here
    ######################################################################
    import sys
    sys.path.append("../")
    from torch_render import Setup_Config_Lightfield
    import torch_render
    import cv2

    test_save_root = "test_cache/"
    os.makedirs(test_save_root,exist_ok=True)
    o3d.io.write_point_cloud(test_save_root+"setup.ply", pcd)

    setup = Setup_Config_Lightfield({"config_dir":args.save_root})

    lumis = np.linspace(0.0,1.0,num=setup.get_light_num())
    lumis = lumis[None,:]
    imgs = torch_render.visualize_lumi(lumis,setup)

    for which_img in range(imgs.shape[0]):
        cv2.imwrite(test_save_root+"lumi_{}.png".format(which_img),imgs[which_img]*255.0)

    mask_states = np.linspace(0.0,1.0,num=setup.get_mask_num())
    mask_states = mask_states[None,:]
    imgs = torch_render.visualize_mask(mask_states,setup)

    for which_img in range(imgs.shape[0]):
        cv2.imwrite(test_save_root+"mask_{}.png".format(which_img),imgs[which_img]*255.0)