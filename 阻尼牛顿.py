import torch
from sympy import *
import numpy as np
import datetime
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"#允许同时加载conda和torch的OpenMP

n=1000
c=100
d=100
精度=1e-6

def 计算某点的梯度和黑塞矩阵的逆(f,x):
	x=torch.tensor(x, requires_grad=True)
	函数值=f(x)
	梯度 = torch.autograd.grad(函数值, x, retain_graph=True, create_graph=True)[0]# 计算一阶导数,因为我们需要继续计算二阶导数,所以创建并保留计算图

	黑塞矩阵 = torch.tensor([],dtype=torch.float64)
	for anygrad in 梯度:  # torch.autograd.grad返回的是元组
		黑塞矩阵 = torch.cat((黑塞矩阵, torch.autograd.grad(anygrad, x, retain_graph=True)[0]))
	黑塞矩阵=黑塞矩阵.view(x.size()[0], -1)

	return 梯度.detach().numpy(),黑塞矩阵.inverse().detach().numpy(),函数值.detach().numpy()#在pytorch 1.8之前是torch.inverse()，1.8是在torch.linalg.inv()

def 数值函数(x):
	函数值=0
	for i in range(n):
	    系数=c+d+1
	    if i==1: 系数=1
	    if i==2: 系数=c+1
	    if i==n-1: 系数=c+d
	    if i==n: 系数=d
	    函数值+=系数*x[i]**2
	return 函数值

自变量们=[]
符号函数=0#目标函数
初始点 = np.array([3.]*n)
步长=Symbol('k')

for i in range(1,n+1):
    当前自变量=Symbol('x'+str(i))#构造一个自变量类
    系数=c+d+1
    if i==1: 系数=1
    if i==2: 系数=c+1
    if i==n-1: 系数=c+d
    if i==n: 系数=d
    符号函数+=系数*当前自变量**2
    自变量们.append(当前自变量)

i=0
d_start = datetime.datetime.now()#开始迭代
while True:
	print('-'*80)
	print(f'{datetime.datetime.now()}：开始第{i+1}次迭代')
	梯度,黑塞矩阵的逆,函数值=计算某点的梯度和黑塞矩阵的逆(数值函数,初始点)
	if((np.sum(np.square(梯度)))**0.5<精度):     break
	方向=-黑塞矩阵的逆@梯度

	目标点s=方向*步长+自变量们
	对应值=[(自变量们[i], 初始点[i]) for i in range(n)]#初始点的对应值列表，给subs函数用
	目标点s=[目标.subs(对应值) for 目标 in 目标点s]

	fk=符号函数.subs([(自变量们[i], 目标点s[i]) for i in range(n)])#用目标点的值替换原目标函数f中的自变向量x。得到关于步长k的一元函数f(k),即求f(k)的最小值s
	k=solve(diff(fk,步长),步长)#对f(k)求导,并使其等于0，求出步长的数值k

	目标点=方向*k[0]+初始点#这是一个向量值
	目标点=np.array([float(目标) for 目标 in 目标点])#sympy.core.numbers.Float转python float
	print("目标点",目标点[5])
	初始点=目标点
	i+=1
d_end = datetime.datetime.now()
print('函数值=',函数值)
print('总耗时',d_end-d_start)