import torch

from sklearn.preprocessing import OneHotEncoder

inp = torch.tensor([1,2,3,4,5],dtype=torch.double)
# print(inp.shape)
inp = torch.unsqueeze(inp,0)
# inp = torch.unsqueeze(inp,0)
# print(inp.shape)
layer = torch.nn.Conv1d(1,3,2,dtype=torch.double)
layer2 = torch.nn.Conv1d(1,3,2,stride=2,dtype=torch.double)
layer3 = torch.nn.Conv1d(3,1,2,dtype=torch.double)
out = layer(inp)
out2 = layer2(inp)  
reqOut = torch.ones((1,3),dtype=torch.double)
reqOut = reqOut.to('cuda')
criterion = torch.nn.MSELoss()

model = torch.nn.Sequential(layer,torch.nn.ReLU(),layer3)
model.zero_grad()
model.to('cuda')
inp = inp.to('cuda')
print(inp.device)

out3 = model(inp)

loss = criterion(out3,reqOut)

print(loss)
loss.backward()

print(inp)
print(out3)
print(loss)

def pytorchTest():
  print(torch.cuda.is_available())

  print(torch.cuda.device_count())

  print(torch.cuda.current_device())

  print(torch.cuda.get_device_name(0))