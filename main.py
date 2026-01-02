import torch

X = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
limit = int(input())

larger_than_limit_sum = (X[X > limit]).sum()

print(larger_than_limit_sum)