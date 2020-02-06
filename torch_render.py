import torch
import numpy as np

def hemi_octa_map(dir):
    '''
    a = [batch,3]
    return = [batch,2]
    '''
    # TODO ADD DIM ASSERT

    p = dir/torch.sum(torch.abs(dir),dim=1,keepdim=True)#[batch,3]
    result = torch.cat([p[:,[0]] - p[:,[1]],p[:,[0]] + p[:, [1]]],dim=1) * 0.5 + 0.5
    return result

def back_hemi_octa_map(a):
    '''
    a = [batch,2]
    return = [batch,3]
    '''
    # TODO ADD DIM ASSERT
    p = (a - 0.5)*2.0
    resultx = (p[:,[0]]+p[:,[1]])*0.5#[batch,1]
    resulty = (p[:,[1]]-p[:,[0]])*0.5#[batch,1]
    resultz = 1.0-torch.abs(resultx)-torch.abs(resulty)
    result = torch.cat([resultx,resulty,resultz],dim=1)#[batch,3]

    return torch.nn.functional.normalize(result,dim=1)

def full_octa_map(dir):
    '''
    dir = [batch,3]
    return=[batch,2]
    '''
    # TODO ADD DIM ASSERT
    p = dir/torch.sum(torch.abs(dir),dim=1,keepdim=True)
    px,py,pz = torch.split(p,[1,1,1],dim=1)
    x,y = px,py

    judgements1 = torch.ge(px,0.0)
    x_12 = torch.where(judgements1,1.0-py,-1.0+py)#[batch,1]
    y_12 = torch.where(judgements1,1.0-px,1.0+px)

    judgements2 = torch.le(px,0.0)
    x_34 = torch.where(judgements2,-1.0-py,1.0+py)
    y_34 = torch.where(judgements2,-1.0-px,-1.0+px)

    judgements3 = torch.ge(py,0.0)
    x_1234 = torch.where(judgements3,x_12,x_34)#[batch,1]
    y_1234 = torch.where(judgements3,y_12,y_34)#[batch,1]

    judgements4 = torch.lt(dir[:,[2]],0.0)
    x = torch.where(judgements4,x_1234,x)
    y = torch.where(judgements4,y_1234,y)

    return (torch.cat([x,y],dim=1)+1.0)*0.5

def back_full_octa_map(a):
    '''
    a = [batch,2]
    return = [batch,3]
    '''
    #TODO ADD DIM ASSERT
    p = a*2.0-1.0
    px,py = torch.split(p,[1,1],dim=1)#px=[batch,1] py=[batch,1]
    x = px#px=[batch,1]
    y = py#px=[batch,1]
    abs_px_abs_py = torch.abs(px)+torch.abs(py)
    
    judgements2 = torch.ge(py,0.0)
    judgements3 = torch.ge(px,0.0)
    judgements4 = torch.le(px,0.0)

    x_12 = torch.where(judgements3,1.0-py,-1.0+py)
    y_12 = torch.where(judgements3,1.0-px,1.0+px)

    x_34 = torch.where(judgements4,-1.0-py,1.0+py)
    y_34 = torch.where(judgements4,-1.0-px,-1.0+px)

    x_1234 = torch.where(judgements2,x_12,x_34)
    y_1234 = torch.where(judgements2,y_12,y_34)

    
    judgements1 = torch.gt(abs_px_abs_py,1)

    resultx = torch.where(judgements1,x_1234,px)#[batch,1]
    resulty = torch.where(judgements1,y_1234,py)#[batch,1]
    resultz = 1.0-torch.abs(resultx)-torch.abs(resulty)
    resultz = torch.where(judgements1,-1.0*resultz,resultz)

    result = torch.cat([resultx,resulty,resultz],dim=1)#[batch,3]

    return torch.nn.functional.normalize(result,dim=1)

def build_frame_f_z(n,theta,device,with_theta=True):
    '''
    n = [batch,3]
    return =t[batch,3] b[batch,3]
    '''
    #TODO ADD DIM ASSERT
    nz = n[:,[2]]
    batch_size = nz.size()[0]

    # try:
    #     constant_001 = build_frame_f_z.constant_001
    #     constant_100 = build_frame_f_z.constant_100
    # except AttributeError:
    #     build_frame_f_z.constant_001 = torch.from_numpy(np.expand_dims(np.array([0,0,1],np.float32),0).repeat(batch_size,axis=0)).to(device)#[batch,3]
    #     build_frame_f_z.constant_100 = torch.from_numpy(np.expand_dims(np.array([1,0,0],np.float32),0).repeat(batch_size,axis=0)).to(device)#[batch,3]
    #     constant_001 = build_frame_f_z.constant_001
    #     constant_100 = build_frame_f_z.constant_100
    
    constant_001 = torch.zeros(batch_size,3,device=device,dtype=torch.float32)
    constant_001[:,2] = 1.0
    constant_100 = torch.zeros(batch_size,3,device=device,dtype=torch.float32)
    constant_100[:,0] = 1.0

    nz_notequal_1 = torch.gt(torch.abs(nz-1.0),1e-6)
    nz_notequal_m1 = torch.gt(torch.abs(nz+1.0),1e-6)

    t = torch.where(nz_notequal_1&nz_notequal_m1,constant_001,constant_100)#[batch,3]

    t = torch.nn.functional.normalize(torch.cross(t,n),dim=1)#[batch,3]
    b = torch.cross(n,t)#[batch,3]

    if not with_theta:
        return t,b

    t = torch.nn.functional.normalize(t*torch.cos(theta)+b*torch.sin(theta),dim=1)

    b = torch.nn.functional.normalize(torch.cross(n,t),dim=1)#[batch,3]
    return t,b

class Torch_Render(object):
    
    def __init__(self,args):
        self.batch_size = args["batch_size"]
        self.if_grey_scale = args["if_grey_scale"]
        self.config_dir = args["config_dir"]
        self.device = args["device"]
        self.load_constants_from_bin(self.config_dir)

    def load_constants_from_bin(self,config_file_dir):
        #load configs
        self.cam_pos = np.fromfile(config_file_dir+"cam_pos.bin",np.float32)
        assert self.cam_pos.shape[0] == 3
        print("[RENDERER]cam_pos:",self.cam_pos)
        
        tmp_data = np.fromfile(config_file_dir+"lights.bin",np.float32).reshape([2,-1,3])
        self.light_poses = tmp_data[0]
        self.light_normals = tmp_data[1]
    
    def get_cam_pos_torch(self,device):
        return torch.from_numpy(self.cam_pos).to(device)
   