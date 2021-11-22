from sympy import *
import numpy as np
import datetime
#测试函数DQDRTIC

n=500
c=100
d=100
精度=1e-6

自变量们=[]
符号函数=0#目标函数

for i in range(1,n+1):
    当前自变量=Symbol('x'+str(i))#构造一个自变量类
    系数=c+d+1
    if i==1: 系数=1
    if i==2: 系数=c+1
    if i==n-1: 系数=c+d
    if i==n: 系数=d
    符号函数+=系数*当前自变量**2
    自变量们.append(当前自变量)

步长=Symbol('k')
初始点=np.array([3.]*n)

i=0#迭代次数
d_start = datetime.datetime.now()#开始迭代

while (True):
    print('-'*80)
    print(f'{datetime.datetime.now()}：开始第{i+1}次迭代')
    对应值=[(自变量们[i], 初始点[i]) for i in range(n)]#初始点的对应值列表，给subs函数用
    print ('i=',i,'f=',符号函数.subs(对应值).evalf())#观察迭代

    梯度们=[diff(符号函数,自变量们[i]).subs(对应值) for i in range(n)]#对各个自变量求偏导，并带入值
    负梯度方向=-np.array(梯度们)

    if((np.sum(np.square(负梯度方向)))**0.5<精度):     break #负梯度的模长达到精度要求时退出

    #用求导的方式，得到步长k
    目标点=负梯度方向*步长+初始点#表达式

    fk=符号函数.subs([(自变量们[i], 目标点[i]) for i in range(n)])#用目标点的值替换原目标函数f中的自变向量x。得到关于步长k的一元函数f(k),即求f(k)的最小值s

    k=solve(diff(fk,步长),步长)#对f(k)求导,并使其等于0，求出步长的数值k

    目标点=负梯度方向*k[0]+初始点#这是一个向量值
    print((np.sum(np.square(目标点-初始点)))**0.5)
    if((np.sum(np.square(目标点-初始点)))**0.5<精度):    break #经过迭代后无明显变化时退出
        
    初始点=目标点#继续迭代
    i=i+1
d_end = datetime.datetime.now()
print('总耗时',d_end-d_start)
