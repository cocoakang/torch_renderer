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
    ###[STEP 0]
    #load constants
    light_normals = setup.get_light_normal_torch()#[lightnum,3]
    light_poses = setup.get_light_poses_torch()#[lightnum,3],
    cam_pos = setup.get_cam_pos_torch()#[3]
    if use_new_cam_pos:
        cam_pos = new_cam_pos
    #rotate object           
    view_mat_model = rotation_axis(rotate_theta,setup.get_rot_axis_torch())#[batch,4,4]
    view_mat_model_t = torch.transpose(view_mat_model,1,2)#[batch,4,4]

    view_mat_for_normal =torch.transpose(torch.inverse(view_mat_model),1,2)#[batch,4,4]
    view_mat_for_normal_t = torch.transpose(view_mat_for_normal,1,2)#[batch,4,4]

    test_node = torch.cat([view_mat_model,view_mat_model_t,view_mat_for_normal,view_mat_for_normal_t],dim=0)
    return None,test_node
    ###[STEP 1]
    ##define input
    with tf.variable_scope("fittinger"):
        self.endPoints[variable_scope_name+"input_parameters"] = input_params
        self.endPoints[variable_scope_name+"positions"] = position
        view_dir = cam_pos - position #shape=[batch,3]
        view_dir = self.normalized(view_dir)#shape=[batch,3]
        self.endPoints[variable_scope_name+"view_dir"] = view_dir

        #build local frame
        frame_t,frame_b = self.build_frame_f_z(view_dir,None,with_theta=False)#[batch,3]
        frame_n = view_dir
        self.endPoints[variable_scope_name+"frame_t"] = frame_t
        self.endPoints[variable_scope_name+"frame_b"] = frame_b
        self.endPoints[variable_scope_name+"frame_n"] = frame_n



    ###[STEP 1.1]
    ###split input parameters into position and others
    if self.if_grey_scale:
        n_2d,theta,ax,ay,pd,ps = tf.split(input_params,[2,1,1,1,1,1],axis=1)
        self.endPoints[variable_scope_name+"pd"] = pd
    else:
        n_2d,theta,ax,ay,pd,ps = tf.split(input_params,[2,1,1,1,3,3],axis=1)

    #position shape=[bach,3]
    # n_2d = tf.clip_by_value(n_2d,0.0,1.0)
    if "n" in use_custom_frame:
        n = global_custom_frame[0]
        if "t" in use_custom_frame:
            t = global_custom_frame[1]
            b = global_custom_frame[2]
        else:
            t,b = self.build_frame_f_z(n,None,with_theta=False)
    else:
        n_local = self.back_hemi_octa_map(n_2d)#[batch,3]
        self.endPoints[variable_scope_name+"normal_local"] = n_local
        t_local,_ = self.build_frame_f_z(n_local,theta,with_theta=True)
        n_local_x,n_local_y,n_local_z = tf.split(n_local,[1,1,1],axis=1)#[batch,1],[batch,1],[batch,1]
        n = n_local_x*frame_t+n_local_y*frame_b+n_local_z*frame_n#[batch,3]
        self.endPoints[variable_scope_name+"normal"]  = n
        t_local_x,t_local_y,t_local_z = tf.split(t_local,[1,1,1],axis=1)#[batch,1],[batch,1],[batch,1]
        t = t_local_x*frame_t+t_local_y*frame_b+t_local_z*frame_n#[batch,3]
        b = tf.cross(n,t)#[batch,3]
    
    if rotate_frame:
        #rotate frame
        pn = tf.expand_dims(tf.concat([n,tf.ones([n.shape[0],1],tf.float32)],axis=1),axis=1)
        pt = tf.expand_dims(tf.concat([t,tf.ones([t.shape[0],1],tf.float32)],axis=1),axis=1)
        pb = tf.expand_dims(tf.concat([b,tf.ones([b.shape[0],1],tf.float32)],axis=1),axis=1)

        n = tf.squeeze(tf.matmul(pn,view_mat_for_normal_t),axis=1)
        t = tf.squeeze(tf.matmul(pt,view_mat_for_normal_t),axis=1)
        b = tf.squeeze(tf.matmul(pb,view_mat_for_normal_t),axis=1)
        n,_ = tf.split(n,[3,1],axis=1)#shape=[batch,3]          
        t,_ = tf.split(t,[3,1],axis=1)#shape=[batch,3]
        b,_ = tf.split(b,[3,1],axis=1)#shape=[batch,3]

    self.endPoints[variable_scope_name+"n"] = n
    self.endPoints[variable_scope_name+"t"] = t
    self.endPoints[variable_scope_name+"b"] = b
    # n = tf.tile(tf.constant([0,0,1],dtype=tf.float32,shape=[1,3]),[self.fitting_batch_size,1])#self.normalized(n)
    # t = tf.tile(tf.constant([0,1,0],dtype=tf.float32,shape=[1,3]),[self.fitting_batch_size,1])#self.normalized(t)
    # ax = tf.clip_by_value(ax,0.006,0.503)
    # ay = tf.clip_by_value(ay,0.006,0.503)
    # pd = tf.clip_by_value(pd,0,1)
    # ps = tf.clip_by_value(ps,0,10)

    #regularizer of params
    # regular_loss_n = self.regularizer_relu(n_2d,1e-3,1.0)+self.regularizer_relu(theta,0.0,math.pi)+self.regularizer_relu(ax,0.006,0.503)+self.regularizer_relu(ay,0.006,0.503)+self.regularizer_relu(pd,1e-3,1)+self.regularizer_relu(ps,1e-3,10)
    # self.endPoints["regular_loss"] = regular_loss_n


    self.endPoints[variable_scope_name+"render_params"] = tf.concat([n,t,b,ax,ay,pd,ps],axis=-1)
    ###[STEP 2]
    ##define rendering
    with tf.variable_scope("rendering"):
        self.endPoints[variable_scope_name+"position_origin"] = position
        if rotate_point:
            position = tf.expand_dims(tf.concat([position,tf.ones([position.shape[0],1],tf.float32)],axis=1),axis=1)
            position = tf.squeeze(tf.matmul(position,view_mat_model_t),axis=1)#position@view_mat_model_t
            position,_ = tf.split(position,[3,1],axis=1)#shape=[batch,3]
        self.endPoints[variable_scope_name+"position_rotated"] = position

        #get real view dir
        view_dir = cam_pos - position #shape=[batch,3]
        view_dir = self.normalized(view_dir)#shape=[batch,3]
        self.endPoints[variable_scope_name+"view_dir_rotated"] = view_dir



        light_poses_broaded = tf.tile(tf.expand_dims(light_poses,axis=0),[self.fitting_batch_size,1,1],name="expand_light_poses")#shape is [batch,lightnum,3]
        light_normals_broaded = tf.tile(tf.expand_dims(light_normals,axis=0),[self.fitting_batch_size,1,1],name="expand_light_normals")#shape is [batch,lightnum,3]
        position_broded = tf.tile(tf.expand_dims(position,axis=1),[1,self.lumitexel_size,1],name="expand_position")
        wi = light_poses_broaded-position_broded
        wi = self.normalized_nd(wi)#shape is [batch,lightnum,3]
        self.endPoints[variable_scope_name+"wi"] = wi


        wi_local = tf.concat([self.dot_ndm_vector(wi,tf.expand_dims(t,axis=1)),
                                self.dot_ndm_vector(wi,tf.expand_dims(b,axis=1)),
                                self.dot_ndm_vector(wi,tf.expand_dims(n,axis=1))],axis=-1)#shape is [batch,lightnum,3]
        
        # view_dir_broaded = tf.tile(tf.expand_dims(view_dir,axis=1),[1,self.lumitexel_size,1])#shape is [batch,lightnum,3]
        wo_local = tf.concat([self.dot_ndm_vector(view_dir,t),
                                self.dot_ndm_vector(view_dir,b),
                                self.dot_ndm_vector(view_dir,n)],axis=-1)#shape is [batch,3]
        wo_local = tf.tile(tf.expand_dims(wo_local,axis=1),[1,self.lumitexel_size,1])#shape is [batch,lightnum,3]

        self.endPoints[variable_scope_name+"wi_local"] = wi_local
        self.endPoints[variable_scope_name+"wo_local"] = wo_local
        
        self.endPoints[variable_scope_name+"light_poses_broaded"] = light_poses_broaded
        self.endPoints[variable_scope_name+"light_normals_broaded"] = light_normals_broaded
        form_factors = self.compute_form_factors(position,n,light_poses_broaded,light_normals_broaded,variable_scope_name,with_cos)#[batch,lightnum,1]
        self.endPoints[variable_scope_name+"form_factors"] = form_factors
        lumi = self.calc_light_brdf(wi_local,wo_local,ax,ay,pd,ps,pd_ps_wanted,specular_component)#[batch,lightnum,channel]
        self.endPoints[variable_scope_name+"lumi_without_formfactor"] = lumi
        
        lumi = lumi*form_factors*1e4*math.pi*1e-2#[batch,lightnum,channel]

        wi_dot_n = self.dot_ndm_vector(wi,tf.expand_dims(n,axis=1))#[batch,lightnum,1]
        lumi = lumi*((tf.sign(wi_dot_n)+1.0)*0.5)
        # judgements = tf.less(wi_dot_n,1e-5)
        # lumi = tf.where(judgements,tf.zeros([self.fitting_batch_size,self.lumitexel_size,1]),lumi)

        n_dot_views = self.dot_ndm_vector(view_dir,n)#[batch,1]
        n_dot_view_dir = tf.tile(tf.expand_dims(n_dot_views,axis=1),[1,self.lumitexel_size,1])#[batch,lightnum,1]
        
        self.endPoints[variable_scope_name+"n_dot_view_dir"] = n_dot_views

        judgements = tf.less(n_dot_view_dir,0.0)
        if self.if_grey_scale:
            rendered_results = tf.where(judgements,tf.zeros([self.fitting_batch_size,self.lumitexel_size,1]),lumi)#[batch,lightnum]
        else:
            rendered_results = tf.where(tf.tile(judgements,[1,1,3]),tf.zeros([self.fitting_batch_size,self.lumitexel_size,3]),lumi)#[batch,lightnum]
        self.endPoints[variable_scope_name+"rendered_results"] = rendered_results

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