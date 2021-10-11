"""
SOURCE: https://stackoverflow.com/questions/53975717/pytorch-connection-between-loss-backward-and-optimizer-step
"""


import torch

x = torch.tensor([1.0], requires_grad=True)
y = torch.tensor([2.0], requires_grad=True)
z = 3*x**2+y**3

print("x.grad: ", x.grad)
print("y.grad: ", y.grad)
# print("z.grad: ", z.grad)

# # print result should be:
# x.grad:  None
# y.grad:  None
# z.grad:  None

# calculate the gradient
z.backward()

print("x.grad: ", x.grad)
print("y.grad: ", y.grad)
# print("z.grad: ", z.grad)

# # print result should be:
# x.grad:  tensor([6.])
# y.grad:  tensor([12.])
# z.grad:  None


# create an optimizer, pass x,y as the paramaters to be update, sutting the learning rate lr=0.1
optimizer = torch.optim.SGD([x, y], lr=0.1)

# executing an update step
optimizer.step()

# print the updated values of x and y
print("x:", x)
print("y:", y)

# print result should be:
# x: tensor([0.4000], requires_grad=True)
# y: tensor([0.8000], requires_grad=True)
