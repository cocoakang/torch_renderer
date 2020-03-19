import torch
import torch.nn as nn
import torch_render
import time
import multiprocessing as mp
from torch.multiprocessing import Process, Queue, Semaphore

class Rendering_Thread(Process):
    def __init__(self,setup,process_id,rendering_configs,device,thread_ctr_map):
        Process.__init__(self)
        print("[PROCESS {}]forked rendering process".format(process_id))
        self.setup = setup
        self.process_id = process_id
        self.rendering_configs = rendering_configs
        self.device = device
        self.need_rendering = False
        self.program_end = False

        self.thread_ctr_map = thread_ctr_map
        self.input_queue = self.thread_ctr_map["input_queue"]
        self.output_queue = self.thread_ctr_map["output_queue"]
        self.device_sph = self.thread_ctr_map["device_sph"]
    
    def run(self):
        print("[PROCESS {}] Starting".format(self.process_id))
        while True:
            input_data = self.input_queue.get()
            # print("[PROCESS {}] Got rendering data".format(self.process_id))
            self.device_sph.acquire()
            # print("[PROCESS {}] Got device".format(self.process_id))
            #data to worker's device
            tmp_input_params = input_data[0].to(self.device,copy=True)
            tmp_input_positions = input_data[1].to(self.device,copy=True)
            tmp_rotate_theta = input_data[2].to(self.device,copy=True)

            del input_data[0]
            del input_data[0]
            del input_data[0]
            del input_data

            #render here
            tmp_lumi,end_points = torch_render.draw_rendering_net(
                self.setup,
                tmp_input_params,
                tmp_input_positions,
                tmp_rotate_theta,
                *self.rendering_configs
            )
            # print("[PROCESS {}] Rendering done".format(self.process_id))
            self.device_sph.release()
            self.output_queue.put([self.process_id,tmp_lumi,end_points]) 
            del tmp_lumi
            # print("[PROCESS {}] Return data done".format(self.process_id))

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
        self.device_sph_list = []
        for which_device in range(self.available_devices_num):
            self.device_sph_list.append(Semaphore(4))

        self.renderer_list = []
        self.input_queue_list = []
        self.output_queue = Queue(self.sample_view_num)
        for which_renderer in range(self.sample_view_num):
            print("[MAIN] create renderer:{}".format(which_renderer))
            tmp_input_queue = Queue()
            self.input_queue_list.append(tmp_input_queue)

            
            cur_device_id = which_renderer%self.available_devices_num
            cur_device = self.available_devices[cur_device_id]
            cur_semaphore = self.device_sph_list[cur_device_id]

            thread_ctr_map = {
                "input_queue":tmp_input_queue,
                "output_queue":self.output_queue,
                "device_sph":cur_semaphore
            }


            tmp_renderer = Rendering_Thread(self.setup,which_renderer,["render_{}".format(which_renderer)],cur_device,thread_ctr_map)
            tmp_renderer.daemon = True
            tmp_renderer.start()
            # print("PID:",tmp_renderer.pid)
            # time.sleep(2.0)
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
        all_param_dim = input_params.size()[1]
        assert all_param_dim == 11 or all_param_dim == 7,"input param dim should be 11 or 7 now:{}".format(all_param_dim)
        channel_num = 3 if all_param_dim == 11 else 1
        ############################################################################################################################
        ##step 2 rendering
        ############################################################################################################################
        rotate_theta_zero = torch.zeros(batch_size,1)
        
        input_params = input_params.to("cpu")
        input_positions = input_positions.to("cpu")
        rotate_theta_zero = rotate_theta_zero.to("cpu")

        for idx in range(self.sample_view_num):
            # print("[MAIN] put data to queue:{}".format(idx))
            self.input_queue_list[idx].put([input_params.detach(),input_positions.detach(),rotate_theta_zero.detach()])
            # time.sleep(3.0)
        del input_params
        del input_positions
        del rotate_theta_zero
        
        result_tensor = torch.empty(self.sample_view_num,batch_size,self.setup.get_light_num(),channel_num,device=origin_device)#(sample_view_num,batchsize,lumilen,channel_num)
        
        for view_id in range(self.sample_view_num):
            tmp_result = self.output_queue.get()
            result_tensor[tmp_result[0]] = tmp_result[1].to(origin_device,copy=True)
            del tmp_result[0]
            del tmp_result[0]
            del tmp_result

        ############################################################################################################################
        ##step 3 grab_all_rendered_result
        ############################################################################################################################
        result_tensor = result_tensor.permute(1,0,2,3)
        
        return result_tensor
