import torch
import torchvision
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn.functional as F
from network import Net
import torchvision.transforms.functional as TF

def main():
  network = Net()
  # network.cuda() # setting để chạy bằng gpu
  # optimizer = optim.SGD(network.parameters(), lr=learning_rate,momentum=momentum)
  network.load_state_dict(torch.load('./results/model.pth'))
  # optimizer.load_state_dict(torch.load('./results/optimizer.pth'))
  network.eval();
  # image = Image.open('test-img.jpg')
  image = Image.open('test_img.jpg').convert('L')
  # plt.figure()
  # plt.imshow(image) 
  # plt.show()  # display it
  x = TF.to_tensor(image)
  x.unsqueeze_(0)
  print(x.shape)
  output = network(x)
  pred = output.data.max(1, keepdim=True)[1]
  print('Predict: ', pred)

if __name__ == '__main__':
  main()