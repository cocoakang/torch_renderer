import torch
import torch.nn as nn
import torch_render
import threading

class Rendering_Thread(threading.Thread):
    def __init__(self,setup,name,rendering_configs,device,thread_ctr_map):
        threading.Thread.__init__(self)
        print("forked rendering process:{}".format(name))
        self.setup = setup
        self.name = name
        self.rendering_configs = rendering_configs
        self.device = device
        self.need_rendering = False
        self.program_end = False

        self.thread_ctr_map = thread_ctr_map
        self.io_list_id = self.thread_ctr_map["io_list_id"]
        self.input_list = self.thread_ctr_map["input_holder"]
        self.output_list = self.thread_ctr_map["output_holder"]

    # def end_of_program(self):
    #     self.program_end = True    
    
    def run(self):
        print("{} Starting".format(self.name))
        while True:
            self.thread_ctr_map["input_sph"].acquire()
            #data to worker's device
            tmp_input_params = self.input_list[self.io_list_id][0].to(self.device)
            tmp_input_positions = self.input_list[self.io_list_id][1].to(self.device)
            tmp_rotate_theta = self.input_list[self.io_list_id][2].to(self.device)

            #render here
            tmp_lumi,end_points = torch_render.draw_rendering_net(
                self.setup,
                tmp_input_params,
                tmp_input_positions,
                tmp_rotate_theta,
                self.name,
                *self.rendering_configs
            )

            self.output_list[self.io_list_id] = [tmp_lumi,end_points]
            self.thread_ctr_map["output_sph"].release()
            # if self.program_end:
            #     break
        # print("{} Done.".format(self.name))

class Multiview_Renderer(nn.Module):
    def __init__(self,args):
        super(Multiview_Renderer,self).__init__()
    
        ########################################
        ##parse configuration                ###
        ########################################
        self.available_devices = args["available_devices"]
        self.available_devices_num = len(self.available_devices)
        self.sample_view_num = args["sample_view_num"]
        TORCH_RENDER_PATH = args["torch_render_path"]

        #######################################
        ## load rendering configs           ###
        #######################################
        standard_rendering_parameters = {
            "config_dir":TORCH_RENDER_PATH+"wallet_of_torch_renderer/blackbox20_render_configs_1x1/"
        }
        self.setup = torch_render.Setup_Config(standard_rendering_parameters)

        #######################################
        ## construct renderer               ###
        #######################################
        self.input_holder = [None]*self.sample_view_num
        self.output_holder = [None]*self.sample_view_num

        self.renderer_list = []
        self.input_sph_list = []
        self.output_sph = threading.Semaphore(0)
        for which_renderer in range(self.sample_view_num):
            input_sph = threading.Semaphore(0)
            self.input_sph_list.append(input_sph)

            thread_ctr_map = {
                "input_sph":input_sph,
                "output_sph":self.output_sph,
                "input_holder":self.input_holder,
                "output_holder":self.output_holder,
                "io_list_id":which_renderer
            }

            cur_device = self.available_devices[which_renderer%self.available_devices_num]
            tmp_renderer = Rendering_Thread(self.setup,"render_{}".format(which_renderer),[],cur_device,thread_ctr_map)
            tmp_renderer.setDaemon(True)
            tmp_renderer.start()
            self.renderer_list.append(tmp_renderer)
        

    def forward(self,input_params,input_positions):
        '''
        input_params=(batch_size,7 or 11) torch tensor
        input_positions=(batch_size,3) torch tensor
        '''
        
        ############################################################################################################################
        ##step 0 unpack batch data
        ############################################################################################################################
        batch_size = input_params.size()[0]
        origin_device = input_params.device
        assert input_positions.size()[0] == batch_size,"input_params shape:{} input_positions shape:{}".format(input_params.size(),input_positions.size())

        ############################################################################################################################
        ##step 2 rendering
        ############################################################################################################################
        rotate_theta_zero = torch.zeros(batch_size,1)
        
        for idx in range(self.sample_view_num):
            self.input_holder[idx] = [input_params,input_positions,rotate_theta_zero]
            self.input_sph_list[idx].release()

        for i in range(self.sample_view_num):
            self.output_sph.acquire()

        ############################################################################################################################
        ##step 3 grab_all_rendered_result
        ############################################################################################################################
        tmp_result_list = [a[0].to(origin_device) for a in self.output_holder]
        result = torch.stack(tmp_result_list,dim=0)#(sample_view_num,lumilen,channel_num)
    
        return result
