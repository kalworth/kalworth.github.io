# JoyRL组队学习笔记(task05)

首先感谢Datawhale组织以及JoyRL教学团队的组队学习以及开源课程！👏👏👏

本次学习活动的课程链接：

https://github.com/datawhalechina/joyrl-book

https://github.com/datawhalechina/easy-rl

## 七、策略梯度方法

### 1. 策略参数化

使用参数为$\theta$的
策略函数$\pi_{\theta}(a|s)$代替策略
$\pi(a|s)$，为了让目标函数
$J(\pi_\theta)$最大化，对目标函数
$-J(\pi_\theta)$求梯度下降来更新策略参数：

$$
\theta \leftarrow \theta - \alpha \nabla_{\theta}(J(\pi_\theta))
$$

### 2. 定义目标函数 $J(\pi_\theta)$

最大化目标函数$J(\pi_\theta)$，核心思想是最大化期望回报。

#### 2.1 基于轨迹推导

一个回合的产生的轨迹如下：

$$
{s_0,a_0,r_0},{s_1,a_1,r_1},{s_2,a_2,r_3},\cdots,{s_T,a_T,r_T}
$$

这条轨迹的概率$\Pr_{\pi}(\tau)$可以根据时间步推导为：

$$
\begin{aligned}
{初始状态s_0}\ \ \Pr_{\pi}(\tau) &= \rho_0(s_0) \\
{采取动作a_0}\ \ \Pr_{\pi}(\tau) &= \rho_0(s_0) \pi_\theta(a_0|s_0)\\
{更新状态s_1}\ \ \Pr_{\pi}(\tau) &= \rho_0(s_0) \pi_\theta(a_0|s_0) P(s_1|s_0, a_0) \\
\cdots \\
\Pr_{\pi}(\tau) &= \rho_0(s_0) \prod_{t=0}^{T-1} \pi_\theta(a_t|s_t) P(s_{t+1}|s_t, a_t)
\end{aligned}
$$

轨迹概率$\Pr_{\pi}(\tau)$可以写为参数
$\theta$的函数：

$$
\Pr_{\pi}(\tau) = p_\theta(\tau)
$$

目标函数可以表示为轨迹概率与回报乘积的积分

$$
J(\pi_\theta)=\int_{\tau} p_\theta(\tau) R(\tau) d \tau = \mathbb{E}_{\tau \sim p_\theta(\tau)}[R(\tau)]
$$

那么目标函数梯度表示为

$$
\nabla_\theta J(\pi_\theta) = \nabla_\theta \int_{\tau} p_\theta(\tau) R(\tau) d \tau = \int_{\tau} \nabla_\theta p_\theta(\tau) R(\tau) d \tau
$$

利用对数函数导数性质以及链式法则，轨迹概率只与参数$\theta$相关，轨迹概率的梯度可以推导表示为

$$
\begin{aligned}
\nabla_\theta \log p_\theta(\tau)&=\frac{\nabla_\theta p_\theta(\tau)}{p_\theta(\tau)} \\
\ \\
\nabla_\theta p_\theta(\tau) &= p_\theta(\tau)\nabla_\theta \log p_\theta(\tau) \\
&=p_\theta(\tau) \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t|s_t)
\end{aligned}
$$

综上目标函数的计算方式为

$$
\begin{aligned}
\nabla_\theta J(\pi_\theta) &= \int_{\tau} \nabla_\theta p_\theta(\tau) R(\tau) d \tau \\
&=\int_{\tau} \nabla_\theta \log \pi_\theta(a_t|s_t) R(\tau) d \tau \\
&= \mathbb{E}_{\tau \sim p_\theta(\tau)}\left[\sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t|s_t) R_t\right]
\end{aligned}
$$

#### 2.2 占用测度推导

占用测度推导从状态价值角度入手，以初始状态的分布$\rho_0$和对应的状态价值$V^{\pi}(s_0)$乘积的积分来表示目标函数

$$
J(\pi)=\int_{s_{0}} \rho_{0}\left(s_{0}\right) V^{\pi}\left(s_{0}\right) d s_{0} = \mathbb{E}_{s_{0} \sim \rho_{0}} \left[V^{\pi}\left(s_{0}\right)\right]
$$

由前文状态价值和动作价值的练习，目标函数可以表述为

$$
J(\pi_\theta)=\int_{s_{0}} \rho_{0}\left(s_{0}\right) \sum_{a} \pi_\theta(a|s_0) Q^{\pi_\theta}\left(s_{0}, a\right) d s_{0}
$$

初始状态分布会影响后续目标函数的计算，不能当作常数项处理，引入平稳分布作为推导前提

平稳分布是在具备不可约和非周期的马尔可夫过程中，系统在长期运行后，初始状态分布是收敛到固定值。平稳分布用$d^{\pi}(s)$表示，那么引入平稳分布后的目标函数表示为

$$
J(\pi_\theta)= \sum_{s} d^{\pi}(s) V^{\pi}(s) = \sum_{s} d^{\pi}(s) \pi_\theta(a|s) Q^{\pi}(s, a) = \mathbb{E}_{s \sim d^{\pi}(s), a \sim \pi_\theta(a|s)}[Q^{\pi}(s, a)]
$$

平稳分布变化缓慢，梯度忽略不计，那么类似的目标函数梯度计算方法经过一系列推导，可得
$$
\begin{aligned}
\nabla_\theta J(\pi_\theta) = \mathbb{E}_{s \sim d^{\pi}(s), a \sim \pi_\theta(a|s)}[\nabla_\theta \log \pi_\theta(a|s) Q^{\pi}(s, a)]
\end{aligned}
$$

**两个角度推导的目标函数梯度等价**

#### 2.3 策略梯度通用表达式

$$
\begin{aligned}
g = \mathbb{E}\left[ \sum_{t=0}^{\infty} \Psi_t \nabla_{\theta} \log \pi_{\theta}(a_t \mid s_t) \right]
\end{aligned}
$$
