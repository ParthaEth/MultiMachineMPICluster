import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.optim as optim

from examples.model.toy_model import ToyModel

def run_training_loop(local_rank, global_rnak, **kwargs):
    # create model and move it to GPU with id rank
    # print(kwargs)
    model = ToyModel().to(local_rank)
    ddp_model = DDP(model)

    loss_fn = torch.nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    total_iterations = 20
    for itr_num in range(total_iterations):
        optimizer.zero_grad()
        outputs = ddp_model(torch.randn(20, 10).to(local_rank))
        labels = torch.randn(20, 5).to(local_rank)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        if global_rnak == 0: # Things that should happen only once such as checkpointing
            print(f'itr:{itr_num}/{total_iterations}. loss: {loss.mean().item()}')