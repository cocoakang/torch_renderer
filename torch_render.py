import torch
import math
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

def build_frame_f_z(n,theta,with_theta=True):
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
    
    constant_001 = torch.zeros_like(n)
    constant_001[:,2] = 1.0
    constant_100 = torch.zeros_like(n)
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

def rotation_axis(t,v,isRightHand=True):
    '''
    t = [batch,1]#rotate rad
    v = [3]#rotate axis(global) 
    return = [batch,4,4]#rotate matrix
    '''
    if isRightHand:
        theta = t
    else:
        print("[RENDERER]Error rotate system doesn't support left hand logic!")
        exit()
    
    batch_size = t.size()[0]

    c = torch.cos(theta)#[batch,1]
    s = torch.sin(theta)#[batch,1]

    m_11 = c + (1-c)*v[0]*v[0]
    m_12 = (1 - c)*v[0]*v[1] - s*v[2]
    m_13 = (1 - c)*v[0]*v[2] + s*v[1]

    m_21 = (1 - c)*v[0]*v[1] + s*v[2]
    m_22 = c + (1-c)*v[1]*v[1]
    m_23 = (1 - c)*v[1]*v[2] - s*v[0]

    m_31 = (1 - c)*v[2]*v[0] - s*v[1]
    m_32 = (1 - c)*v[2]*v[1] + s*v[0]
    m_33 = c + (1-c)*v[2]*v[2]

    tmp_zeros = torch.zeros_like(t)
    tmp_ones = torch.ones_like(t)

    res = torch.cat([
        m_11,m_12,m_13,tmp_zeros,
        m_21,m_22,m_23,tmp_zeros,
        m_31,m_32,m_33,tmp_zeros,
        tmp_zeros,tmp_zeros,tmp_zeros,tmp_ones
    ],dim=1)

    res = res.view(-1,4,4)
    return res

def compute_form_factors(position,n,light_poses,light_normals,end_points,with_cos=True):
    '''
    position = [batch,3]
    n = [batch,3]
    light_poses = [lightnum,3]
    light_normals = [lightnum,3]

    with_cos: if this is true, form factor adds cos(ldir.light_normals)  

    return shape=[batch,lightnum,1]
    '''
    ldir = torch.unsqueeze(light_poses,dim=0)-torch.unsqueeze(position,dim=1)#[batch,lightnum,3]
    dist = torch.sqrt(torch.sum(ldir**2,dim=2,keepdim=True))#[batch,lightnum,1]
    ldir = torch.nn.functional.normalize(ldir,dim=2)#[batch,lightnum,3]

    static_zero = torch.zeros(1,device=n.device)

    a = torch.max(torch.sum(ldir*torch.unsqueeze(n,dim=1),dim=2,keepdim=True),static_zero)#[batch,lightnum,1]
    if not with_cos:
        return a
    b = dist*dist#[batch,lightnum,1]
    c = torch.max(torch.sum(ldir*torch.unsqueeze(light_normals,dim=0),dim=2,keepdim=True),static_zero)#[batch,lightnum,1]
    # r_2_cos = b/(c+1e-6)
    # cos_r_2 = c/b
    # self.endPoints[variable_scope_name+"r_2_cos"] = r_2_cos
    # self.endPoints[variable_scope_name+"cos_r_2"] = cos_r_2
    # self.endPoints[variable_scope_name+"cos2"] = a
    return a/(b+1e-6)*c

def ggx_G1_aniso_honntai(v,vz,ax,ay):
    axayaz = torch.cat([ax,ay,torch.ones_like(ax)],dim=1)#[batch,3]
    vv = v*torch.unsqueeze(axayaz,dim=1)#[batch,lightnum,3]
    # return 2.0/(1.0+(self.norm(vv)/(vz+1e-6)))
    return 2.0*vz/(vz+torch.norm(vv,dim=2,keepdim=True)+1e-6)#[batch,lightnum,1]

def ggx_G1_aniso(v,ax,ay,vz):
    '''
    v = [batch,lightnum,3]
    ax = [batch,1]
    ay = [batch,1]
    vz = [batch,lightnum,1] 
    return shape = [batch,lightnum,1]
    '''
    return torch.where(torch.le(vz,torch.zeros_like(vz)),torch.zeros_like(vz),ggx_G1_aniso_honntai(v,vz,ax,ay))#[batch,lightnum,1]

    # comparison = (tf.sign(vz)+1.0)*0.5
    # hontai = ggx_G1_aniso_honntai(v,vz,ax,ay)
    # return hontai*comparison

def ggx_brdf_aniso(wi,wo,ax,ay,specular_component):
    '''
    wi = [batch,lightnum,3]
    wo = [batch,3]
    ax = [batch,1]
    ay = [batch,1]
    return shape = [batch,lightnum,1]
    '''
    static_one = torch.ones(1,device=wi.device,dtype=torch.float32)
    static_zero = torch.zeros_like(static_one)

    wo = torch.unsqueeze(wo,dim=1).repeat(1,wi.size()[1],1)#[batch,lightnum,3]

    wi_z = wi[:,:,[2]]#tf.expand_dims(tf.gather(wi,indices=2,axis=2,name="wi_z"),axis=-1)#shape=[batch,lightnum,1]
    wo_z = wo[:,:,[2]]#tf.expand_dims(tf.gather(wo,indices=2,axis=2,name="wo_z"),axis=-1)#shape=[batch,lightnum,1]
    denom = 4*wi_z*wo_z#shape=[batch,lightnum,1]
    vhalf = torch.nn.functional.normalize(wi+wo,dim=2)#[batch,lightnum,3]
    tmp = torch.min(torch.max(1.0-torch.sum(wi*vhalf,dim=2,keepdim=True),static_zero),static_one)#[batch,lightnum,1]
    F0 = 0.04
    F = F0+(1-F0)* tmp * tmp * tmp * tmp * tmp#[batch,lightnum,1]
    
    axayaz = torch.unsqueeze(torch.cat([ax,ay,torch.ones_like(ax)],dim=1),dim=1)#[batch,1,3]
    vhalf = vhalf/(axayaz+1e-6)#[batch,lightnum,3]
    vhalf_norm = torch.norm(vhalf,dim=2,keepdim=True)#[batch,lightnum,1]
    length = vhalf_norm*vhalf_norm##[batch,lightnum,1]
    D = 1.0/(math.pi*torch.unsqueeze(ax,dim=1)*torch.unsqueeze(ay,dim=1)*length*length)#[batch,lightnum,1]

    judgement_wiz_less_equal_0 = torch.le(wi_z,static_zero)
    judgement_woz_less_equal_0 = torch.le(wo_z,static_zero)

    tmp_ones = torch.ones_like(denom)
    safe_denom = torch.where(judgement_wiz_less_equal_0,tmp_ones,denom)
    safe_denom = torch.where(judgement_woz_less_equal_0,tmp_ones,safe_denom)

    tmp = tmp_ones
    if "D" in specular_component:
        tmp = tmp * D
    if "F" in specular_component:
        tmp = tmp * F
    if "G" in specular_component:
        tmp = tmp * ggx_G1_aniso(wi,ax,ay,wi_z)*ggx_G1_aniso(wo,ax,ay,wo_z)
    if "B" in specular_component:
        tmp = tmp / (safe_denom+1e-6) 
    
    #[batch,lightnum,1]

    
    tmp_zeros = torch.zeros_like(tmp)


    res = torch.where(judgement_wiz_less_equal_0,tmp_zeros,tmp)
    res = torch.where(judgement_woz_less_equal_0,tmp_zeros,res)

    # wi_z_sign = (tf.sign(wi_z)+1.0)*0.5#shape=[batch,lightnum,1]
    # wo_z_sign = (tf.sign(wo_z)+1.0)*0.5#shape=[batch,lightnum,1]
    
    # # res = tmp*wi_z_sign*wo_z_sign
    # self.endPoints["4"] = wi_z_sign
    # self.endPoints["5"] = wo_z_sign
    # self.endPoints["6"] = denom+1e-6
    return res

def calc_light_brdf(wi_local,wo_local,ax,ay,pd,ps,pd_ps_wanted,specular_component):
    '''
    wi_local = [batch,lightnum,3]
    wo_local = [batch,3]
    ax = [batch,1]
    ay = [batch,1]
    pd = [batch,channel]
    ps = [batch,channel]
    return shape=[batch,lightnum,channel]
    '''
    if pd_ps_wanted == "both":
        b = ggx_brdf_aniso(wi_local,wo_local,ax,ay,specular_component)#[batch,lightnum,1]
        ps = torch.unsqueeze(ps,dim=1)#[batch,1,channel]
        a = torch.unsqueeze(pd/math.pi,dim=1)#[batch,1,1]
        return a+b*ps
    elif pd_ps_wanted =="pd_only":
        a = torch.unsqueeze(pd/math.pi,dim=1)#[batch,1,1]
        return a.repeat(1,wi_local.size()[1],1)
    elif pd_ps_wanted == "ps_only":
        b = ggx_brdf_aniso(wi_local,wo_local,ax,ay,specular_component)#[batch,lightnum,1]
        ps = torch.unsqueeze(ps,dim=1)#[batch,1,channel]
        return b*ps
    # return b*ps# return a+b*ps

def draw_rendering_net(setup,input_params,position,rotate_theta,variable_scope_name,
    with_cos = True,pd_ps_wanted="both",rotate_point = True,specular_component="D_F_G_B",
    global_custom_frame=None,use_custom_frame="",rotate_frame=True,new_cam_pos=None,use_new_cam_pos=False):
    '''
    setup is Setup_Config class
    input_params = (rendering parameters) shape = [self.fitting_batch_size,self.parameter_len] i.e.[24576,10]
    position = (rendering positions) shape=[self.fitting_batch_size,3]
    variable_scope_name = (for variable check a string like"rendering1") 
    rotate_theta = [self.fitting_batch_size,1]
    return shape = (rendered results)[batch,lightnum,1] or [batch,lightnum,3]
    specular_component means the degredient of brdf(B stands for bottom)
    "D_F_G_B"


    with_cos: if True,lumitexl is computed with cos and dir
    '''
    end_points = {}
    batch_size = input_params.size()[0]
    ###[STEP 0]
    #load constants
    light_normals = setup.get_light_normal_torch()#[lightnum,3]
    light_poses = setup.get_light_poses_torch()#[lightnum,3],
    light_num = light_poses.size()[0]
    cam_pos = setup.get_cam_pos_torch()#[3]
    if use_new_cam_pos:
        cam_pos = new_cam_pos
    #rotate object           
    view_mat_model = rotation_axis(rotate_theta,setup.get_rot_axis_torch())#[batch,4,4]
    view_mat_model_t = torch.transpose(view_mat_model,1,2)#[batch,4,4]

    view_mat_for_normal =torch.transpose(torch.inverse(view_mat_model),1,2)#[batch,4,4]
    view_mat_for_normal_t = torch.transpose(view_mat_for_normal,1,2)#[batch,4,4]

    test_node = torch.cat([view_mat_model,view_mat_model_t,view_mat_for_normal,view_mat_for_normal_t],dim=0)

    ###[STEP 1] define frame
    view_dir = cam_pos - position #shape=[batch,3]
    view_dir = torch.nn.functional.normalize(view_dir,dim=1)#shape=[batch,3]

    ###[STEP 1.1]
    ###split input parameters into position and others
    if input_params.size()[1] == 7:
        n_2d,theta,ax,ay,pd,ps = torch.split(input_params,[2,1,1,1,1,1],dim=1)
    elif input_params.size()[1] == 11:
        n_2d,theta,ax,ay,pd,ps = torch.split(input_params,[2,1,1,1,3,3],dim=1)
    else:
        print("[RENDER ERROR] error param len!")
        exit(-1)
    #position shape=[bach,3]
    # n_2d = tf.clip_by_value(n_2d,0.0,1.0)
    if "n" in use_custom_frame:
        n = global_custom_frame[0]
        if "t" in use_custom_frame:
            t = global_custom_frame[1]
            b = global_custom_frame[2]
        else:
            t,b = build_frame_f_z(n,None,with_theta=False)
    else:
         #build local frame
        frame_t,frame_b = build_frame_f_z(view_dir,None,with_theta=False)#[batch,3]
        frame_n = view_dir#[batch,3]

        n_local = back_hemi_octa_map(n_2d)#[batch,3]
        t_local,_ = build_frame_f_z(n_local,theta,with_theta=True)
        n = n_local[:,[0]]*frame_t+n_local[:,[1]]*frame_b+n_local[:,[2]]*frame_n#[batch,3]
        t = t_local[:,[0]]*frame_t+t_local[:,[1]]*frame_b+t_local[:,[2]]*frame_n#[batch,3]
        b = torch.cross(n,t)#[batch,3]

    if rotate_frame:
        #rotate frame
        static_tmp_ones = torch.ones(batch_size,1,dtype=torch.float32,device=n.device)
        pn = torch.unsqueeze(torch.cat([n,static_tmp_ones],dim=1),1)#[batch,1,4]
        pt = torch.unsqueeze(torch.cat([t,static_tmp_ones],dim=1),1)#[batch,1,4]
        pb = torch.unsqueeze(torch.cat([b,static_tmp_ones],dim=1),1)#[batch,1,4]

        n = torch.squeeze(torch.matmul(pn,view_mat_for_normal_t),1)[:,:3]#[batch,1,4]
        t = torch.squeeze(torch.matmul(pt,view_mat_for_normal_t),1)[:,:3]
        b = torch.squeeze(torch.matmul(pb,view_mat_for_normal_t),1)[:,:3]
    
    if rotate_point:
        position = torch.unsqueeze(torch.cat([position,static_tmp_ones],dim=1),dim=1)#[batch,1,4]
        position = torch.squeeze(torch.matmul(position,view_mat_model_t),dim=1)[:,:3]#shape=[batch,3]
        
    ###[STEP 2]
    ##define rendering

    #get real view dir
    view_dir = torch.unsqueeze(cam_pos,dim=0) - position #shape=[batch,3]
    view_dir = torch.nn.functional.normalize(view_dir,dim=1)#shape=[batch,3]

    # light_poses_broaded = tf.tile(tf.expand_dims(light_poses,axis=0),[self.fitting_batch_size,1,1],name="expand_light_poses")#shape is [batch,lightnum,3]
    # light_normals_broaded = tf.tile(tf.expand_dims(light_normals,axis=0),[self.fitting_batch_size,1,1],name="expand_light_normals")#shape is [batch,lightnum,3]
    wi = torch.unsqueeze(light_poses,dim=0)-torch.unsqueeze(position,dim=1)#[batch,lightnum,3]
    wi = torch.nn.functional.normalize(wi,dim=2)#shape is [batch,lightnum,3]


    wi_local = torch.cat([  torch.sum(wi*torch.unsqueeze(t,dim=1),dim=2,keepdim=True),
                            torch.sum(wi*torch.unsqueeze(b,dim=1),dim=2,keepdim=True),
                            torch.sum(wi*torch.unsqueeze(n,dim=1),dim=2,keepdim=True)],dim=2)#shape is [batch,lightnum,3]
    
    wo_local = torch.cat([  torch.sum(view_dir*t,dim=1,keepdim=True),
                            torch.sum(view_dir*b,dim=1,keepdim=True),
                            torch.sum(view_dir*n,dim=1,keepdim=True)],dim=1)#shape is [batch,3]
    
    
    form_factors = compute_form_factors(position,n,light_poses,light_normals,end_points,with_cos)#[batch,lightnum,1]

    lumi = calc_light_brdf(wi_local,wo_local,ax,ay,pd,ps,pd_ps_wanted,specular_component)#[batch,lightnum,channel]
    
    lumi = lumi*form_factors*1e4*math.pi*1e-2#[batch,lightnum,channel]

    wi_dot_n = torch.sum(wi*torch.unsqueeze(n,dim=1),dim=2,keepdim=True)#[batch,lightnum,1]
    # lumi = lumi*((tf.sign(wi_dot_n)+1.0)*0.5)
    lumi = torch.where(torch.lt(wi_dot_n,1e-5),torch.zeros_like(lumi),lumi)#[batch,lightnum,channel]

    n_dot_views = torch.sum(view_dir*n,dim=1,keepdim=True)#[batch,1]
    n_dot_view_dir = torch.unsqueeze(n_dot_views,dim=1).repeat(1,light_num,1)#tf.tile(tf.expand_dims(n_dot_views,axis=1),[1,self.lumitexel_size,1])#[batch,lightnum,1]

    rendered_results = torch.where(torch.lt(n_dot_view_dir,0.0),torch.zeros_like(lumi),lumi)#[batch,lightnum]

    return rendered_results

class Setup_Config(object):
    
    def __init__(self,args):
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

        self.rot_axis = np.array([0.0,0.0,1.0],np.float32)# TODO read from calibration file
    
    def get_cam_pos_torch(self):
        try:
            return self.cam_pos_torch
        except AttributeError:
            self.cam_pos_torch = torch.from_numpy(self.cam_pos).to(self.device)
            return self.cam_pos_torch
   
    def get_light_normal_torch(self):
        try:
            return self.light_normals_torch
        except AttributeError:
            self.light_normals_torch = torch.from_numpy(self.light_normals).to(self.device)
            return self.light_normals_torch
    
    def get_light_poses_torch(self):
        try:
            return self.light_poses_torch
        except AttributeError:
            self.light_poses_torch = torch.from_numpy(self.light_poses).to(self.device)
            return self.light_poses_torch
    
    def get_rot_axis_torch(self):
        try:
            return self.rot_axis_torch
        except AttributeError:
            self.rot_axis_torch = torch.from_numpy(self.rot_axis).to(self.device)
            return self.rot_axis_torch