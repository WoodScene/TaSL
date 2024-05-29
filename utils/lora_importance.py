import math
import torch

from typing import Optional, List 

class RankAllocator(object):
    """
    Args:
        model: the model that we apply AdaLoRA to.
        init_warmup (`int`): The steps of initial fine-tuning warmup.
        beta1 (`float`): The hyperparameter of EMA for sensitivity smoothing.
        beta2 (`float`): The hyperparameter of EMA for undertainty quantification.
        total_step (`int`): The total training steps, correctly configured before training.
        tb_writter (`SummaryWriter`): Tensorboard SummaryWriter. 
        tb_writter_loginterval (`int`): The logging interval of SummaryWriter. 
    """
    def __init__(
        self, model, 
        init_warmup:int, 
        beta1:float, 
        beta2:float, 
        total_step:Optional[int]=None, 
        tb_writter=None,
        tb_writter_loginterval:int=500, 
    ):

        self.initial_warmup = init_warmup
        self.beta1 = beta1
        self.beta2 = beta2
        self.total_step = total_step

        self.model = model
        self.ipt = {} 
        self.exp_avg_ipt = {}
        self.exp_avg_unc = {}
        self.cat_ipt = {}
        self.rank_pattern = {} 
        self.get_lora_param_name()

        self.tb_writter = tb_writter
        self.log_interval = tb_writter_loginterval 

        assert (self.beta1<1 and self.beta1>0)
        assert (self.beta2<1 and self.beta2>0)

    def set_total_step(self, total_step:int): 
        # Set total step number 
        self.total_step = total_step
        # self.initial_warmup = int(self.total_step / 10) + 1
        self.initial_warmup = 0
        print(f"total_step is {self.total_step}, initial_warmup is {self.initial_warmup}")
        assert self.total_step>self.initial_warmup


    def get_lora_param_name(self):
        # Prepare the budget scheduler 
        self.name_set = set() 
        self.total_rank = 0 
        self.shape_dict = {}
        for n,p in self.model.named_parameters():
            if "lora_A" in n: 
                name_mat = n.replace("lora_A", "%s")
                self.name_set.add(name_mat)
                self.total_rank += p.size(0) 
                self.shape_dict[n] = p.shape
            if "lora_B" in n:
                self.shape_dict[n] = p.shape
        self.name_set = list(sorted(self.name_set)) 
        # print(f"name set is {self.name_set}")


    def update_ipt(self, model, global_step): 
        for n,p in model.named_parameters():
            if "lora_" in n: 
                if n not in self.ipt:
                    self.ipt[n] = torch.zeros_like(p)
                    self.exp_avg_ipt[n] = torch.zeros_like(p) 
                    self.exp_avg_unc[n] = torch.zeros_like(p) 
                with torch.no_grad():
                    # Calculate sensitivity 
                    self.ipt[n] = (p * p.grad).abs().detach()
                    # Update sensitivity 
                    self.exp_avg_ipt[n] = self.beta1 * self.exp_avg_ipt[n] + \
                                        (1-self.beta1)*self.ipt[n]
                    # Update uncertainty 
                    self.exp_avg_unc[n] = self.beta2 * self.exp_avg_unc[n] + \
                                        (1-self.beta2)*(self.ipt[n]-self.exp_avg_ipt[n]).abs()
                # print(f"step is {global_step}")
                # print(f"name is {n}")
                # print(f"ipt is {self.ipt[n]}")
                # print()
                

    def calculate_score(self, p=None, metric="ipt"):
        assert len(self.exp_avg_ipt) == len(self.exp_avg_unc)
        ipt_name_list = []
        ipt_score_list = []
        for n in self.exp_avg_ipt:
            print(f"name is {n}")
            ipt_name_list.append(n)
            if metric == "ipt":
                # Combine the senstivity and uncertainty 
                ipt_score = self.exp_avg_ipt[n] * self.exp_avg_unc[n]
            elif metric == "mag":
                ipt_score = p.abs().detach().clone() 
            else:
                raise ValueError("Unexcptected Metric: %s"%metric)
            
            # print(f"score is {ipt_score}")
            ipt_score_mean = torch.mean(ipt_score).item()
            print(f"mean score is {ipt_score_mean}")
            ipt_score_list.append(ipt_score_mean)
        # print(f"ipt is {self.ipt}")
        # print(f"exp_avg_ipt is {self.exp_avg_ipt[n]}")
        # print(f"exp_avg_unc is {self.exp_avg_unc[n]}")
        return ipt_name_list, ipt_score_list

    def _combine_ipt(self, ipt_E, ipt_AB):
        ipt_AB = ipt_AB.sum(dim=1, keepdim=False)
        sum_ipt = ipt_E.view(-1) + ipt_AB.view(-1)
        return sum_ipt
    
    def update_score(self, model, global_step):
        if global_step < self.total_step and global_step > self.initial_warmup:
            # Update importance scores element-wise 
            self.update_ipt(model, global_step)
        # print(f"ipt is {self.ipt}")
        # print(f"exp_avg_ipt is {self.exp_avg_ipt}")
        # print(f"exp_avg_unc is {self.exp_avg_unc}")
        # sys.exit(1)

    

