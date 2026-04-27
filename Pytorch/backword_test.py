import torch  
  
# 创建一个tensor并设置requires_grad=True来追踪其计算历史  
x = torch.ones(2, 2, requires_grad=True)  
  
# 对这个tensor做一次运算：  
y = x + 2  
  
# y是计算的结果，所以它有grad_fn属性  
print(y.grad_fn)  
  
# 对y进行更多的操作  
z = y * y * 3  
out = z.mean()  
  
print(z, out)  
  
# 使用.backward()来进行反向传播，计算梯度  
out.backward()  
  
# 输出梯度d(out)/dx  
print(x.grad)  