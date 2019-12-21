import torch
import torchvision
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from network import Net
from sklearn.metrics import precision_recall_fscore_support as score

# chuẩn bị dữ liệu =========================
n_epochs = 5 # Số lần lặp
batch_size_train = 128 # training size
batch_size_test = 1000 # testing data
learning_rate = 0.05
momentum = 0.5
log_interval = 10

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)
# =========================================

# tải dữ liệu =============================
train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('./files/', train=True, download=True,
    transform=torchvision.transforms.Compose([
      torchvision.transforms.ToTensor(),
      torchvision.transforms.Normalize(
        (0.1307,), (0.3081,))
    ])),
  batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('./files/', train=False, download=True,
    transform=torchvision.transforms.Compose([
      torchvision.transforms.ToTensor(),
      torchvision.transforms.Normalize(
        (0.1307,), (0.3081,))
    ])),
  batch_size=batch_size_test, shuffle=True)
# ==========================================

# khởi tạo mạng =============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# network = Net()
network = Net().to(device=device)
optimizer = optim.SGD(network.parameters(), lr=learning_rate,momentum=momentum)
# network.cuda()
# ==========================================

# các biến để kiểm tra tiến độ =============
train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]
# ==========================================

# train ====================================
def train(epoch):
  network.train()
  for batch_idx, (data, target) in enumerate(train_loader):
    target = target.to(device)
    data = data.to(device)
    optimizer.zero_grad()
    output = network(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.step()
    if batch_idx % log_interval == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(train_loader.dataset),
        100. * batch_idx / len(train_loader), loss.item()))
      train_losses.append(loss.item())
      train_counter.append(
        (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
      torch.save(network.state_dict(), './results/model.pth')
      torch.save(optimizer.state_dict(), './results/optimizer.pth')
# ===================================================================

# test ============================================================
# precisionArr = []
# recallAr = []
# fscoreAr = []
# lossArr = []
# accuracyArr = []
def test():
  network.eval()
  test_loss = 0
  correct = 0
  predicted = [] 
  y_test = [] 
  with torch.no_grad():
    for data, target in test_loader:
      target = target.to(device)
      data = data.to(device)
      output = network(data)
      test_loss += F.nll_loss(output, target, size_average=False).item()
      pred = output.data.max(1, keepdim=True)[1]
      correct += pred.eq(target.data.view_as(pred)).sum()
      predicted = np.concatenate((predicted, pred.cpu().detach().numpy().flatten()), axis=0)
      y_test = np.concatenate((y_test, target.data.view_as(pred).cpu().detach().numpy().flatten()), axis=0)
  test_loss /= len(test_loader.dataset)
  test_losses.append(test_loss)

  precision, recall, fscore, support = score(y_test, predicted, average='macro')
  print('precision: {}'.format(precision))
  print('recall: {}'.format(recall))
  print('fscore: {}'.format(fscore))
  print('support: {}'.format(support))
  print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))
# ==============================================================

# chạy chương trình ==============================================
test() # chạy test trước khi lặp để khởi tạo model với những tham số ngẫu nhiên
for epoch in range(1, n_epochs + 1):
  train(epoch)
  test()

fig = plt.figure()
plt.plot(train_counter, train_losses, color='blue')
plt.scatter(test_counter, test_losses, color='red')
plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
plt.xlabel('number of training examples seen')
plt.ylabel('negative log likelihood loss')
plt.show()
