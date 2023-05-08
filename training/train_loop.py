import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.optim as optim


from model.toy_model import ToyModel

def run_training_loop(local_rank):
    # create model and move it to GPU with id rank
    model = ToyModel().to(local_rank)
    ddp_model = DDP(model, device_ids=[local_rank])

    loss_fn = torch.nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    total_iterations = 20
    for itr_num in range(total_iterations):
        optimizer.zero_grad()
        outputs = ddp_model(torch.randn(20, 10))
        labels = torch.randn(20, 5).to(local_rank)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        print(f'itr:{itr_num}/{total_iterations}. loss: {loss.mean().item()}')