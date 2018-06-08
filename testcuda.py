import torch
with torch.cuda.device(1):
	a=torch.randn(1,2,3)
	b=torch.rand(1,2,3)
	c=a+b
	print(c)
