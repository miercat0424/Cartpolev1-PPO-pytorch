import torch 

a = torch.randn(4)
print(a)
print(torch.clamp(a, min=-0.5, max=0.5))

min = torch.linspace(-1, 1, steps=4)
print(torch.clamp(a, min=min))

# tensor([-1.4861, -0.9035,  0.3934, -0.8129])
# tensor([-0.5000, -0.5000,  0.3934, -0.5000])
# tensor([-1.0000, -0.3333,  0.3934,  1.0000])