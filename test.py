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
  # network.load_state_dict(torch.load('./results/model.pth'))
  network.load_state_dict(torch.load('./res-valid/model.pth'))
  network.eval();
  # image = Image.open('test-img.jpg')
  image = Image.open('test_img.jpg').convert('L')
  # plt.figure()
  # plt.imshow(image) 
  # plt.show()  # display it
  x = TF.to_tensor(image)
  x.unsqueeze_(0)
  output = network(x)
  pred = output.data.max(1, keepdim=True)[1]
  print('Predict: ', pred.cpu().detach().numpy()[0][0])

if __name__ == '__main__':
  main()
