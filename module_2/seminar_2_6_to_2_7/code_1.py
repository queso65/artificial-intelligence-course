import torch

w =  torch.tensor([[5. ,10. ],[1. ,2. ]], requires_grad=True)

function =  torch.prod(torch.log(torch.log(w + 7)))
function.backward()

print(w.grad) # Код для самопроверки