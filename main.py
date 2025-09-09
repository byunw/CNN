import torchvision
from torchvision.transforms import ToTensor

train_dataset = torchvision.datasets.CIFAR10(root='data/',train=True,transform=ToTensor())
test_dataset = torchvision.datasets.CIFAR10(root='data/',train=False,transform=ToTensor())

#test code
assert train_dataset.data.size == 153600000
assert test_dataset.data.size == 30720000

assert train_dataset.data[0][:,:,0].size == 1024 #train_dataset.data[0][:,:,0] has pixel values of the first layer of the first image
assert train_dataset.data[0][:,:,1].size == 1024 #train_dataset.data[0][:,:,1] has pixel values of the second layer of the first image
assert train_dataset.data[0][:,:,2].size == 1024 #train_dataset.data[0][:,:,2] has pixels values of the third layer of the first image

assert train_dataset.data[1][:,:,0].size == 1024 #train_dataset.data[1][:,:,0] has pixel values of the first layer of the second image
assert train_dataset.data[1][:,:,1].size == 1024 #train_dataset.data[1][:,:,1] has pixel values of the second layer of the second image
assert train_dataset.data[1][:,:,2].size == 1024 #train_dataset.data[1][:,:,2] has pixel values of the third layer of the second image
#test code

train_dataset_classes = train_dataset.classes

#image1 has normalized values
image1,label1 = train_dataset[0]
assert train_dataset.data[0][:,:,0][1][1] == 0
assert train_dataset.data[0][:,:,0][0][30] == 152































