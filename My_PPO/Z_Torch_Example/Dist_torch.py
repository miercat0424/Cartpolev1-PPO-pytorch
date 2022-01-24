import torch as T
import torch.nn.functional as F
from torch.distributions.categorical import Categorical as C

action_logits   = T.rand(5)
action_probs    = F.softmax(action_logits, dim=-1)                          # SoftMax: Sum = 1
print(action_logits, action_probs, sep="\n")

dist = C(action_probs)
for i in range(10): 
    action = dist.sample()
    print(action)
    print(dist.log_prob(action), T.log(action_probs[action]),sep="\n")      # Totally Same Thing



