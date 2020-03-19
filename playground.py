import os
import time
import torch
import torch.multiprocessing as mp
import psutil

class Consumer(mp.Process):

    def __init__(self, queue):
        super(Consumer, self).__init__()
        self.queue = queue       
        self.device = torch.device("cuda:1")


    def run(self):
        print('Consumer: ', os.getpid())
        while True:        
            tensor = self.queue.get()
            copied_tensor = tensor.to(self.device,copy=True)
            del tensor

            process = psutil.Process()
            print('Consumer mem: ', process.memory_info().rss, end='\r')

class Producer(mp.Process):

    def __init__(self, queue):
        super(Producer, self).__init__()
        self.queue = queue
        self.device = torch.device('cpu')        

    def run(self):
        print('Producer: ', os.getpid())
        while True:        
            tensor = torch.ones([2, 4], dtype=torch.float32, device=self.device)
            self.queue.put(tensor)
            time.sleep(0.001)


if __name__ == '__main__':
    print(mp.get_all_sharing_strategies())

    queue = mp.Queue()

    consumer = Consumer(queue)
    producer = Producer(queue)

    consumer.start()
    producer.start()    

    consumer.join()
    producer.join()