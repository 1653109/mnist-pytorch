import torch
import torchvision
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import csv
from network import Net
from sklearn.metrics import precision_recall_fscore_support as score
from torch.utils.data.sampler import SubsetRandomSampler
import os.path

# chuẩn bị dữ liệu =========================
n_epochs = 20 # Số lần lặp
batch_size_train = 64 # training size
batch_size_test = 1000 # testing data
learning_rate = 0.01
momentum = 0.5
log_interval = 10

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)
# =========================================

# tải dữ liệu =============================
my_data = torchvision.datasets.MNIST('./files/', train=True, download=True,
  transform=torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
      (0.1307,), (0.3081,))
  ]))

test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('./files/', train=False, download=True,
    transform=torchvision.transforms.Compose([
      torchvision.transforms.ToTensor(),
      torchvision.transforms.Normalize(
        (0.1307,), (0.3081,))
    ])),
  batch_size=batch_size_test, shuffle=True)
# khởi tạo tập train và tập validation từ train_loader =====
## chia ngẫu nhiên tập train_loader
validation_plit = 0.1;
val_len = int(np.floor(validation_plit * len(my_data)))
indices = list(range(len(my_data)))
validation_idx = np.random.choice(indices, size=val_len, replace=False)
train_idx = list(set(indices) - set(validation_idx))
## đẩy dữ liệu vào các tập train và validate
train_sampler = SubsetRandomSampler(train_idx)
validation_sampler = SubsetRandomSampler(validation_idx)

train_loader = torch.utils.data.DataLoader(my_data, sampler=train_sampler, batch_size=batch_size_train)
validation_loader = torch.utils.data.DataLoader(my_data, sampler=validation_sampler, batch_size=batch_size_test)
data_loaders = {"train": train_loader, "val": validation_loader}
data_lengths = {"train": len(train_idx), "val": val_len}
# ==========================================

# khởi tạo mạng =============================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
network = Net().to(device=device)
optimizer = optim.SGD(network.parameters(), lr=learning_rate,momentum=momentum)
# ==========================================

epoch_next = 1 #lưu giá trị vòng lặp, nếu mới bắt đầu thì = 1, nếu train tiếp thì đọc checkpoint để lấy
# các biến để kiểm tra tiến độ =============
train_losses = []
train_counter = []

valid_precisions = []
valid_recalls = []
valid_fscores = []
valid_accuracies = []
valid_losses = []

test_losses = []
test_precisions = []
test_recalls = []
test_fscores = []
test_accuracies = []
test_counter = [i*data_lengths['train'] for i in range(n_epochs + 1)]
# ==========================================

# kiểm tra checkpoint để train tiếp, nếu có
if os.path.isfile("./results/checkpoint.pth"):
  checkpoint = torch.load('./results/checkpoint.pth')
  network.load_state_dict(checkpoint['model_state_dict'])
  optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

  epoch_next = checkpoint['epoch']
  train_losses = checkpoint['train_losses']
  train_counter = checkpoint['train_counter']

  valid_precisions = checkpoint['valid_precisions']
  valid_recalls = checkpoint['valid_recalls']
  valid_fscores = checkpoint['valid_fscores']
  valid_accuracies = checkpoint['valid_accuracies']
  valid_losses = checkpoint['valid_losses']

  test_losses = checkpoint['test_losses']
  test_precisions = checkpoint['test_precisions']
  test_recalls = checkpoint['test_recalls']
  test_fscores = checkpoint['test_fscores']
  test_accuracies = checkpoint['test_accuracies']
  test_counter = checkpoint['test_counter']

# train ====================================
def train(epoch):
  valid_loss = 0
  correct = 0
  predicted = [] 
  y_test = []
  for phase in ['train', 'val']:
    if phase == 'train':
      network.train()
    else:
      network.eval()
    for batch_idx, (data, target) in enumerate(data_loaders[phase]):
      target = target.to(device)
      data = data.to(device)
      optimizer.zero_grad()
      output = network(data)
      
      if phase == 'train':
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
      
      if phase == 'val':
        valid_loss += F.nll_loss(output, target, size_average=False).item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum()
        predicted = np.concatenate((predicted, pred.cpu().detach().numpy().flatten()), axis=0)
        y_test = np.concatenate((y_test, target.data.view_as(pred).cpu().detach().numpy().flatten()), axis=0)

      if batch_idx % log_interval == 0:
        if phase == 'train':
          print('Train Epoch: {} [{}/{} ({:.4f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), data_lengths['train'],
            100. * batch_idx / len(data_loaders['train']), loss.item()))
          train_losses.append(loss.item())
          train_counter.append((batch_idx*64) + ((epoch-1)*data_lengths['train']))
          torch.save(network.state_dict(), './results/model.pth')
          torch.save(optimizer.state_dict(), './results/optimizer.pth')
          torch.save({
            'epoch': epoch,
            'model_state_dict': network.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': train_losses,
            'train_counter': train_counter,
            'valid_precisions': valid_precisions,
            'valid_recalls' : valid_recalls,
            'valid_fscores': valid_fscores,
            'valid_accuracies': valid_accuracies,
            'valid_losses': valid_losses,
            'test_losses': test_losses,
            'test_precisions': test_precisions,
            'test_recalls': test_recalls,
            'test_fscores': test_fscores,
            'test_accuracies': test_accuracies,
            'test_counter': test_counter
            }, './results/checkpoint.pth')

    if phase == 'val':
      valid_loss /= data_lengths['val']
      precision, recall, fscore, support = score(y_test, predicted, average='macro')
      valid_losses.append(valid_loss)
      valid_accuracies.append(correct.item() / data_lengths['val'])
      valid_precisions.append(precision)
      valid_recalls.append(recall)
      valid_fscores.append(fscore)
      print('\nValidation set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        valid_loss, correct, data_lengths['val'],
        100. * correct / data_lengths['val']))
      
# ===================================================================

# test ============================================================
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
  test_precisions.append(precision)
  test_recalls.append(recall)
  test_fscores.append(fscore)
  test_accuracies.append(100. * correct / len(test_loader.dataset))
  print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))
# ==============================================================

# chạy chương trình ==============================================
# test() # chạy test trước khi lặp để khởi tạo model với những tham số ngẫu nhiên
for epoch in range(epoch_next, n_epochs + 1):
  train(epoch)
  # test()
test()

with open('results.csv', 'w', newline='') as file:
  writer = csv.writer(file)
  writer.writerow(['Loss', 'Accuracy', 'Precision', 'Recalls', 'F1 score'])
  for epoch in range(0, n_epochs):
    writer.writerow([valid_losses[epoch], valid_accuracies[epoch], valid_precisions[epoch], valid_recalls[epoch], valid_fscores[epoch]])

plt.plot(range(1, n_epochs + 1), valid_losses, color='green')
plt.plot(range(1, n_epochs + 1), valid_accuracies, color='red')
plt.plot(range(1, n_epochs + 1), valid_precisions, color='blue')
plt.plot(range(1, n_epochs + 1), valid_recalls, color='purple')
plt.plot(range(1, n_epochs + 1), valid_fscores, color='black')
plt.legend(['Loss', 'Accuracies', 'Precision', 'Recalls', 'F1 score'], loc='upper right')
plt.xlabel('n_epochs')
plt.ylabel('%')
plt.title('Graph')
plt.show()
