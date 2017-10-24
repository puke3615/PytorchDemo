import tensorflow.examples.tutorials.mnist.input_data as input_data
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch
import numpy as np
from torch.autograd import Variable

mnist = input_data.read_data_sets('/tmp/data/mnist', one_hot=True)


class MnistModel(nn.Module):
    def __init__(self):
        super(MnistModel, self).__init__()
        self.fc1 = nn.Linear(784, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = F.softmax(x)
        return x


MAX_ITERATOR = 100
BATCH_SIZE = 128

model = MnistModel()
optimizer = optim.Adam(model.parameters(), 1e-2)


def V(numpy):
    return Variable(torch.from_numpy(numpy).float())


print 'Start training...'
for step in range(1, MAX_ITERATOR + 1):
    images, labels = mnist.train.next_batch(BATCH_SIZE)
    # labels = labels.astype(np.int64)
    optimizer.zero_grad()
    outputs = model(V(images))
    criterion = nn.MSELoss()
    loss = criterion(outputs, V(labels))
    loss.backward()
    optimizer.step()
    if step % 10 == 0 or step == MAX_ITERATOR:
        print 'Step %s, Loss %s' % (step, loss.data[0])

print '\nStart test...'
test_images, test_labels = mnist.test.images, mnist.test.labels
correct = 0
for image, label in zip(test_images, test_labels):
    outputs = model(V(image))
    pred = outputs.data.max(0, keepdim=True)[1]
    correct += pred.eq(torch.from_numpy(label).max(0, keepdim=True)[1])
accuracy = correct.double() / len(test_labels)
print 'Accuracy is %s%%' % (100. * accuracy.data)
