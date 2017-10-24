import torch
from torch.autograd import Variable

x = Variable(torch.ones(2), requires_grad=True)

z = 4 * x * x

y = z.norm()

print y.data

y.backward()
print x.grad
