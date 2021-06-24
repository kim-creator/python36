(∂f∂x0,∂f∂x1)을 벡터로 묶어서 표현

최소지점으로 가기 위한 x0의 방향과 크기, 최소지점으로 가기 위한 x1의 방향과 크기

기울기가 크면 클 수록 그 만큼 많이 움직여야 한다.

기울기가 기리키는 쪽은 각 장소에서(좌표에서) 함수의 출력값 x20+x21의 결괏값이 가장 작은 곳

⭐️⭐️⭐️ 경사하강법(경사법), Gradient Descent ⭐️⭐️⭐️

기울기를 줄여 나가는 방법

최초 지점에서 시작해서 일정 거리만큼 이동하면서 기울기를 수정(갱신)

어디로 이동해요? 함수의 값이 최소지점이 되는 방향으로 일정 거리만큼 움직인다.

경사하강법의 원리

현 위치에서 기울어진 방향으로 일정 거리만큼 이동

갱신 되는 위치(좌표)가 일정한게 아니고, 미분 값을 보고 갱신해야 할 수치를 일정하게 조정

학습률 ( learning rate η )이라고 한다.

이동한 곳에서도 미분을 통해 기울기를 구하고, 기울기를 구한 방향으로 이동

이 과정을 최솟점을 찾는 지점까지 반복(step)

경사하강법의 수식

x0=x0−η∂f∂x0

x1=x1−η∂f∂x1

η : Learning Rate ( 하이퍼 파라미터 )

# 경사하강법 구현

# f : 경사하강법을 수행할 함수 ( 미분 대상 함수 )

# init_x : x의 최초 지점

# lr : learning rate

# step_num : 경사하강법 수행 횟수

def gradient_descent(f, init_x, lr=0.01, step_num=100):

  x = init_x

​

  for i in range(step_num):

    # 1. 기울기 배열 구하기

    grads = numerical_gradient(f, x)

    print("좌표 : {} / 기울기 : {}".format(x, grads))

​

    # 2. 경사하강법 공식을 이용한 좌표 갱신

    x = x - lr*grads

​

  return x

ef function_2(x):

  return np.sum(x**2)

start_x = np.array([-3.0, 4.0])

gradient_descent(function_2, start_x, lr=0.1)

import numpy as np

import matplotlib.pylab as plt

​

​

def gradient_descent(f, init_x, lr=0.01, step_num=100):

    x = init_x

    x_history = []

​

    for i in range(step_num):

        x_history.append( x.copy() )

​

        grad = numerical_gradient(f, x)

        x -= lr * grad

​

    return x, np.array(x_history)

​

init_x = np.array([-3.0, 4.0])    

​

lr = 0.1

step_num = 20

x, x_history = gradient_descent(function_2, init_x, lr=lr, step_num=step_num)

​

plt.plot( [-5, 5], [0,0], '--b')

plt.plot( [0,0], [-5, 5], '--b')

plt.plot(x_history[:,0], x_history[:,1], 'o')

​

plt.xlim(-3.5, 3.5)

plt.ylim(-4.5, 4.5)

plt.xlabel("X0")

plt.ylabel("X1")

plt.show()

**학습률(Learning Rate)이 너무 크거나 작으면??**

​

# 학습률이 너무 클 때. lr=10.0

start_x = np.array([-3.0, 4.0])

result, _ = gradient_descent(function_2, start_x, lr=10.0)

print("Learning Rate 10.0 : {}".format(result))

​

기울기가 너무 크면 발산한다.

​

# 학습률이 너무 작은 예

start_x = np.array([-3.0, 4.0])

result, _ = gradient_descent(function_2, start_x, lr=1e-10)

print("Learning Rate 1e-10 : {}".format(result))

​

기울기가 너무 작으면 최소지점으로 가지 못하고 갱신이 끝난다.

​

# 신경망에서의 Gradient Descent

* 손실값(Loss)을 최소로 하는 모델 파라미터를 구하는 과정

* 이 과정을 **최적화(Optimization)**

* 신경망에서의 모델 파라미터는?

* **가중치(W)와 편향(b)**

* Loss를 최소화 하기위한 가중치와 편향을 구하는 과정을 **최적화**라고 한다.

* 경사하강법은 여러 최적화 기법의 일종

​

**신경망의 학습이란?**

* Loss($L$)값을 최소화 시키는 가중치($W$)와 편향($b$)을 구한다.

* 미분은 어떻게 쓰일까?

* $\frac{\partial L}{\partial W}$

* $\frac{\partial L}{\partial b}$

​

$$

W=\begin{pmatrix} w_{11} & w_{21} & w_{31} \\ w_{12}&w_{22}&w_{32} \\\end{pmatrix}, b = \left (b_1, b_2, b_3 \right )

$$

​

$$

\frac{\partial L}{\partial W}=\begin{pmatrix} \frac{\partial L}{\partial w_{11}} & \frac{\partial L}{\partial w_{21}} & \frac{\partial L}{\partial w_{31}} \\ \frac{\partial L}{\partial w_{12}}&\frac{\partial L}{\partial w_{22}}&\frac{\partial L}{\partial w_{32}} \\\end{pmatrix}, \frac{\partial L}{\partial b} = \left (\frac{\partial L}{\partial b_1}, \frac{\partial L}{\partial b_2}, \frac{\partial L}{\partial b_3} \right )

$$

​

**가중치 W, b를 갱신하기 위한 경사하강법 공식**

$$

W = W-\eta \frac{\partial L}{\partial W}, b = b - \eta \frac{\partial L}{\partial b}

$$

​

# 실습에 필요한 함수들

import numpy as np

​

def sigmoid(x):

return 1 / (1 + np.exp(-x))

​

def softmax(x):

if x.ndim == 2:

x = x.T

x = x - np.max(x, axis=0)

y = np.exp(x) / np.sum(np.exp(x), axis=0)

return y.T 

​

x = x - np.max(x) # 오버플로 대책

return np.exp(x) / np.sum(np.exp(x))

​

def cross_entropy_error(y, t):

if y.ndim == 1:

t = t.reshape(1, t.size)

y = y.reshape(1, y.size)

​

# 훈련 데이터가 원-핫 벡터라면 정답 레이블의 인덱스로 반환

if t.size == y.size:

t = t.argmax(axis=1)

​

batch_size = y.shape[0]

return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

​

def numerical_gradient(f, x):

h = 1e-4 # 0.0001

grad = np.zeros_like(x)

​

it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])

while not it.finished:

idx = it.multi_index

tmp_val = x[idx]

x[idx] = float(tmp_val) + h

fxh1 = f(x) # f(x+h)

​

x[idx] = tmp_val - h 

fxh2 = f(x) # f(x-h)

grad[idx] = (fxh1 - fxh2) / (2*h)

​

x[idx] = tmp_val # 값 복원

it.iternext() 

​

return grad

​

## SimpleNet 만들기

* 입력을 두개 받는 `[x1, x_2]` 3개의 뉴런을 가진 신경망

* 편향 고려 x, 가중치만 사용

​

# 신경망 클래스

class SimpleNet:

​

# 초기화에서는..?

# 신경망이 초기에 가지고 있어야 할 매개변수를 세팅

# 신경망 매개변수 초기화 작업을 생성자인 __init__ 메소드에서 수행

def __init__(self):

# 1) 정규분포 랜덤 * 0.01 사용

# 2) 카이밍 히 초깃값 ( He 초깃값 ) - ReLU를 Activation Function으로 사용할 때 사용하는 초기화 방식

# 3) 사비에르 초깃값(글로로트 초깃값) ( Xavier 초깃값 ) - Sigmoid를 Activation Function으로 사용할 때

​

self.W = np.random.randn(2, 3) # (1)번 방식

​

def predict(self, x):

return x @ self.W

​

def loss(self, x, t):

# 손실(loss)을 구할 때 필요한 것

# 예측값(y), 정답(t), loss 함수( cross entropy error )

z = self.predict(x)

y = softmax(z)

​

loss = cross_entropy_error(y, t)

return loss

​

net = SimpleNet()

print("가중치 : \n{}".format(net.W))

​

x = np.array([0.6, 0.9])

p = net.predict(x)

​

print("예측값 : {}".format(p))

​

# loss 구해보기

t = np.array([1, 0, 0])

t_error = np.array([0, 1, 0])

​

print("정답을 잘 예측 했을 때의 Loss : {:.3f}".format(net.loss(x, t)))

print("정답을 잘못 예측 했을 때의 Loss : {:.3f}".format(net.loss(x, t_error)))

​

$$

\frac{\partial L}{\partial W}=\begin{pmatrix} \frac{\partial L}{\partial w_{11}} & \frac{\partial L}{\partial w_{21}} & \frac{\partial L}{\partial w_{31}} \\ \frac{\partial L}{\partial w_{12}}&\frac{\partial L}{\partial w_{22}}&\frac{\partial L}{\partial w_{32}} \\\end{pmatrix}

$$

​

# 1. Loss 구하기 위한 함수

# 2. W에 대한..

# dL / dW

​

# net.loss를 미분할 함수를 따로 만든 것임 (W에 대한...)

def f(W):

return net.loss(x, t)

​

# 혹은

# loss_W = lambda W : net.loss(x, t)

​

# Loss를 구하는 함수 f에 대한 모든 W들의 기울기를 구할 수 있다.

# W의 각 원소에 대한 편미분이 수행된다.

​

dW = numerical_gradient(f, net.W)

print(dW)
