# 强化学习（Reinforcement Learning）

## 简介

### 机器学习的分类
- 监督学习
- 非监督学习
- 强化学习

### 强化学习的提出
- 我们如何认识这个世界？

举个例子， 一个小孩看到一堆火
![img](img/rl-intro-fire-1.png)

他可以靠近火
![img](img/rl-intro-fire-2.png)

也可以摸一摸这把火
![img](img/rl-intro-fire-3.png)

小孩子不同的行为结果是不一样的，当他靠近火的时候，他得到了温暖，感到舒适，是一种奖励。而当他真的摸一把的时候会烫到手，感到疼痛，是一种惩罚。这就是我们生活中学习的过程，强化学习是参考这个过程发明的机器学习方法。

- 智能体和环境
强化学习中有两个对象，
一个是智能体（Agent），就是我们要训练的智能
一个是环境（Environment ），就是智能体要跟他交互的环境的总和。

环境有各种状态
智能体可以采取不同的行动
在不同的环境状态下，智能体采取不同的行动会有不同的结果，就像我们上面的例子下孩子靠近火或者摸火就是两个不同的行动。反过来，智能体的行动也会使环境变成另外的状态。

智能体不同行动所带来的结果我们用`奖励`来描述。像上面的例子，靠近火的时候奖励是`+1`,摸火的时候奖励是`-1`,即不靠近火也不摸火的时候奖励是`0`。

- 奖励
所以智能体学习的过程可以归结为奖励最大化的优化问题。

- 强化学习过程
![img](img/rl-intro-agent.png）

- 奖励最大化
这就是为什么在强化学习中，为了获得最佳行为，我们需要最大化预期的累积奖励。每个时间步t的累积奖励可定义为：
![img](img/rl-intro-reward-1.png)
相当于：
![img](img/rl-intro-reward-2.png)

- 折扣奖励
想一想我们身边的例子，我们在做决策的时候总是考虑短期利益和长期利益，短期利益和长期利益不一定是统一的，这时候就要在短期利益和长期利益之间做平衡，增强学习在实现的时候用的是折扣，就是给长期利益打折。
为了给奖励打折，可以这样做：
我们定义了一个名为gamma的折扣率。它必须介于0和1之间。伽玛越大，折扣越小。这意味着学习，agent更关心长期奖励。
另一方面，伽玛越小，折扣越大。这意味着我们的 agent 更关心短期奖励。
累积的折扣预期奖励是：

![img](img/rl-intro-reward-3.png)

- Exploration/Exploitation trade off (探索/利用 权衡)
在研究解决强化学习问题的不同策略之前，我们必须涵盖另一个非常重要的主题：`探索/利用` 权衡。
探索是寻找有关环境的更多信息。
利用是利用已知信息来最大化奖励。
请记住，我们的RL智能体的目标是最大化预期的累积奖励。但是，我们可能会陷入局部陷阱。
![img](img/rl-intro-exploration-exploitation.png)
在这个游戏中，我们的鼠标可以有无限量的小奶酪（每个+1）。但在迷宫的顶端有一大片奶酪（+1000）。
但是，如果只专注于奖励，agent永远不会达到巨大的奶酪。它只会得到最近的奖励来源，即使这个来源很小（利用）。
但如果我们的智能体进行了一些探索，那么它就有可能获得巨大的回报。
这就是我们所说的勘探/开采权衡。我们必须定义一个有助于处理这种权衡的规则。我们将在以后的文章中看到处理它的不同方法。
生活中这种权衡其实也是不剩枚举的。。。

### 分类

- 情节性任务 或 连续性任务 （Episodic or Continuing tasks）

- 基于价值
- 基于策略
- 基于模型

### 方法
- 学习和规划 （Learning & Planning）
- 探索和利用 （Exploration & Exploitation）
- 预测和控制 （Prediction & Control）

### 马尔可夫过程


## Q-Learning -- Q学习

## Deep Q-Learning -- 深度Q学习

## Policy Gradients -- 策略梯度

## A2C （Advantage Actor Critic） & A3C （Asynchronous Advantage Actor Critic）

## PPO （Proximal Policy Optimization） -- 近端策略优化

## Curiosity-Driven learning -- 好奇心驱动的学习

## 让我们动手吧
[Git -- Deep Reinforcement Learning Course](https://github.com/simoninithomas/Deep_reinforcement_learning_Course)

## Reference

- [简介篇（第一部分）](https://www.yanxishe.com/TextTranslation/1390)
- [通过Q学习进行强化学习（第二部分）](https://www.yanxishe.com/TextTranslation/1394)
- [以Doom为例一文带你读懂深度Q学习（第三部分）](https://www.yanxishe.com/TextTranslation/1395)
- [以 Cartpole 和 Doom 为例介绍策略梯度 （第四部分）](https://www.yanxishe.com/TextTranslation/1406)
- [简单介绍A3C （第五部分）](https://www.yanxishe.com/TextTranslation/1407)
- [以刺猬索尼克游戏为例讲解PPO（第六部分）](https://www.yanxishe.com/TextTranslation/1408)
- [好奇心驱动的学习](https://www.yanxishe.com/TextTranslation/1188)
- [Markov Decision Process(马尔科夫决策过程)](https://zhuanlan.zhihu.com/p/35354956)
- [Bellman Equation(贝尔曼方程)](https://zhuanlan.zhihu.com/p/35261164)

- [《强化学习》第一讲 简介](https://zhuanlan.zhihu.com/p/28084904)
- [《强化学习》第二讲 马尔科夫决策过程](https://zhuanlan.zhihu.com/p/28084942)
- [《强化学习》第三讲 动态规划寻找最优策略](https://zhuanlan.zhihu.com/p/28084955)
- [《强化学习》第四讲 不基于模型的预测](https://zhuanlan.zhihu.com/p/28107168)
- [《强化学习》第五讲 不基于模型的控制](https://zhuanlan.zhihu.com/p/28108498)
- [《强化学习》第六讲 价值函数的近似表示](https://zhuanlan.zhihu.com/p/28223841)
- [《强化学习》第七讲 策略梯度](https://zhuanlan.zhihu.com/p/28348110)
- [《强化学习》第八讲 整合学习与规划](https://zhuanlan.zhihu.com/p/28423255)
- [《强化学习》第九讲 探索与利用](https://zhuanlan.zhihu.com/p/28428947)
