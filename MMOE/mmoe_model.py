import torch

from torch import nn

class MMOE(nn.Module):
    def __init__(self, num_experts, num_tasks, num_inputs, num_neurons_expert, num_neurons_tower, drop_rate=0.2):
        super().__init__()
        self.num_experts = num_experts
        self.num_tasks = num_tasks
        self.num_inputs = num_inputs
        self.num_neurons_expert = num_neurons_expert
        self.num_neurons_tower = num_neurons_tower
        ## Experts
        for i in range(num_experts):
            setattr(self, 'expert'+str(i), nn.Sequential(
                nn.Linear(num_inputs, num_neurons_expert),
                nn.ReLU(),
                nn.Dropout(p=drop_rate)
            ))
        ## Gates
        for i in range(num_tasks): # number of towers, fixed to 2.
            setattr(self, 'gate'+str(i), nn.Sequential(
                nn.Linear(num_inputs, num_experts),
                nn.Softmax(dim=1)
            ))
        ## Towers
        for i in range(num_tasks):
            setattr(self, 'tower'+str(i), nn.Linear(num_neurons_expert, num_neurons_tower))

    def forward(self, xv):
        bs = xv.shape[0]
        ## experts
        out_experts = torch.zeros(self.num_experts, bs, self.num_neurons_expert)
        for i in range(self.num_experts):
            out_experts[i] = getattr(self, 'expert'+str(i))(xv)
        ## gates and weighted opinions
        input_towers = torch.zeros(self.num_tasks, bs, self.num_neurons_expert)
        for i in range(self.num_tasks):
            gate = getattr(self, 'gate'+str(i))(xv)
            for j in range(self.num_experts):
                input_towers[i] += gate[:,j].unsqueeze(dim=1)*out_experts[j]
        ## towers
        out_towers = torch.zeros(self.num_tasks, bs, self.num_neurons_tower)
        for i in range(self.num_tasks):
            out_towers[i] = getattr(self, 'tower'+str(i))(input_towers[i])
        output = torch.sigmoid(out_towers)
        # return out_towers
        return output[0], output[1]

class MMOE_T1(nn.Module):
    def __init__(self, num_experts, num_tasks, num_inputs, num_neurons_expert, num_neurons_tower, drop_rate=0.2):
        super().__init__()
        self.num_experts = num_experts
        self.num_tasks = num_tasks
        self.num_inputs = num_inputs
        self.num_neurons_expert = num_neurons_expert
        self.num_neurons_tower = num_neurons_tower
        ## Experts
        for i in range(num_experts):
            setattr(self, 'expert'+str(i), nn.Sequential(
                nn.Linear(num_inputs, num_neurons_expert),
                nn.ReLU(),
                nn.Dropout(p=drop_rate)
            ))
        ## Gate
        self.gate0 = nn.Sequential(
                nn.Linear(num_inputs, num_experts),
                nn.Softmax(dim=1)
            )
        ## Tower
        self.tower0 = nn.Linear(num_neurons_expert, num_neurons_tower)

    def forward(self, xv):
        bs = xv.shape[0]
        ## experts
        out_experts = torch.zeros(self.num_experts, bs, self.num_neurons_expert)
        for i in range(self.num_experts):
            out_experts[i] = getattr(self, 'expert'+str(i))(xv)
        ## gates and weighted opinions
        input_towers = torch.zeros(bs, self.num_neurons_expert)
        gate = self.gate0(xv)
        for j in range(self.num_experts):
            input_towers += gate[:,j].unsqueeze(dim=1)*out_experts[j]
        ## towers
        # out_towers = torch.zeros(bs, self.num_neurons_tower)
        out_towers = self.tower0(input_towers)
        # for i in range(self.num_tasks):
        #     out_towers[i] = getattr(self, 'tower'+str(i))(input_towers[i])
        # return out_towers
        output = torch.sigmoid(out_towers)
        return output