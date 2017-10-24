import torch
import time

a = torch.FloatTensor(2, 2)

print 2. * a

last = time.time()
result = torch.zeros(2, 2)
for i in range(int(1e6)):
    result += a
print result
offset = (time.time() - last)
print 'Take %.2f s.' % offset
