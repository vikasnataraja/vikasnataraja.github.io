---
layout: post
title: Basics of Reinforcement Learning
subtitle: Q-function, policies and rewards
cover-img: /assets/img/basics_rl/rl_cover.jpeg
tags: [rl, robotics, ai-series]
---

Welcome to my very first blog on this revamped website! I decided to write this blog about Reinforcement Learning because it is an area of Artificial Intelligence that I am absolutely fascinated by! From teaching robots how to pick up cans from a table to playing Mario, the applications of RL are limitless. It is a highly active area of research (see Deepmind, Covariant AI) and there are still many things to be explored and discovered. And so in this post, I will attempt to explain the basics of RL and we will gradually move to more complicated concepts in other blog posts.

# What is Reinforcement Learning?

Simply put, Reinforcement Learning is where an agent in an environment is trying to maximize or optimize the way it behaves to achieve a certain *reward*. And as the name suggests, we use positive and negative reinforcements to teach the agent how to behave (i.e. what actions to take at states). The overall objective of the agent is to maximize the reward and needs to learn how to do that. That was a lot of fancy words and terms so let's back up for a second and break it down.

<p><strong>But wait, what is this "agent" and what is this "reward"?</strong></p>

To answer that question, let's consider an example:

You're playing fetch with your dog and every time she brings the ball back to you, you give her a treat. Not that you should play this game for long because you would run out of treats pretty soon but in this scenario, the dog is the agent. She knows that every time she's successful in bringing the ball back to you, she gets a treat. So you could say, the agent (the dog) is trying to get the maximum reward (treat) by taking some action (fetching the ball).

<figure align="center">
  <img width="500" height="300" src="/assets/img/basics_rl/dog_fetch.jpg" alt="Dog Reinforcement Learning">
    <figcaption> Dog playing fetch :) Source: KDNuggets</figcaption>
</figure>

Mathematically, we can represent it as $r_t$ referring to reward at timestep $t$.

Now that we have some notion of a reward, let's move on to a couple of extended topics: discount factor and discounted future cumulative rewards. Now, stay with me here, it may sound complicated but it's really not. To explain these terms, let's take another example - chess. In chess, the goal is to surround the opponent and make it impossible to escape. In other words, checkmate. To that end, you make moves (taking actions) that can perhaps result in you losing a pawn or another chess piece but may allow you to eventually win the game. So, in a sense, you are playing the long-game and thinking ahead even if it's only a couple of moves ahead. In RL, that's what the discount factor $\gamma$ is for - to decide how much importance should be given to future rewards compared to immediate rewards. Usually, the $\gamma$ value is set $0<\gamma<1$. Setting $\gamma=1$ means the agent will evaluate each of its actions based on the sum total of all its future rewards. [This StackOverflow answer](https://stackoverflow.com/a/54346760/12623546) does very well to explain the difference $\gamma=0.9$ makes compared to $\gamma=0.99$. We will cover this in a later blog with some demonstrations in OpenAI Gym.

Using this discount factor $\gamma$ in our rewards, we can now describe our discounted future cumulative words as the reward that takes into account the discount factor. Mathematically, we use the discount factor exponentially as $\gamma^k$ to indicate that it affects the reward exponentially:

\begin{equation}
\label{eq:reward_future_cumulative}
  R_t = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + ... = \sum_{k=0}^{\infty} \gamma^k r_{t+k+1} \nonumber
\end{equation}

So, all this equation is saying is the farther away a reward is, scale that reward appropriately. If we choose $\gamma=1$, then we weigh all future rewards equally. Choosing $\gamma=0.9$ weighs the rewards in the immediate future heavier than the rewards in the distant future. Depending on the environment we're in, we tune our discount factor differently.

<figure align="center">
  <img width="500" height="550" src="/assets/img/basics_rl/rewards_meme.jpg" alt="Future rewards meme">
</figure>

<p><strong>Policies</strong></p>
Policy is how the agent knows what action to take at different states. It describes how to act in a state by using probabilities. If at a state $s$, the probability of going left is 0.4 and the probability of going right is 0.6, then mathematically, we can write that as $\pi(s,left) = 0.4$ and $\pi(s,right) = 0.6$. This is describing the probability of taking some action $a$ at state $s$. It is providing a kind of mapping to go from state to action which is why it is also represented as $\pi(a|s,\theta)$ to indicate that we need to find an action given a state a meaning it is a stochastic policy. The $\theta$ is a parameter (or a vector of parameters) to fine-tune the learning because each state may have multiple actions and we need to find the best action for each state. Since this is a probability, the summation of all probabilities should be 1 i.e. $\sum_{a} \pi(s,a) = 1$. But that was an easy instance and in an environment where there may be tens or hundreds of states each with multiple actions, it becomes important to learn the policy and this can be achieved in many different ways and we will cover that in a later blog post.

<p><strong>Value functions</strong></p>
To learn about the correct actions to take, we need an expression that characterizes how good the quality of the possible actions are. Simply put, we need something to say *"this action is good"* or *"this action is not so good"*. If we do that, we can then simply say, take the action that results in the highest value out of those possibilities because remember, in RL, we don't have a direct control over which states we end up in but rather we control our actions to make sure we end up in a certain state. But to actually know which action is good or bad, we use value functions that can quantify the quality of an action. Remember that $R_t$ is the future cumulative reward given by: $R_t = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + ... $. Using that, we can describe value functions as:

\begin{equation}
\label{eq:value_function}
  V_\pi(s) = \mathbb{E}_\pi [R_t|S_t = s] \nonumber
\end{equation}

\begin{equation}
\label{eq:q_function}
  Q_\pi(s,a) = \mathbb{E}_\pi [R_t|S_t = s, A_t = a] \nonumber
\end{equation}

Here, the $V(s)$ function is called the **state-value** because it is the expected return if we are in state $s$ at time $t$, $S_t=s$ and the $Q(s,a)$ function is called the **action-value** of a state-action pair which describes the value of taking an action $a$ at state $s$ at time $t$. But, we still need to go integrate the immediate rewards and the future discounted rewards and we do this by using the **Bellman Equations**. Let's break down the value functions to understand the Bellman equations:

\begin{array}{lcl}
  V_\pi(s) & = & \mathbb{E}_\pi [R_t|S_t = s] \newline
  & = & \mathbb{E}\_\pi [R\_{t+1} + \gamma R\_{t+2} + \gamma^2 R\_{t+3} + \gamma^3 R\_{t+4} + ...|S\_t = s] \newline
  & = & \mathbb{E}\_\pi [R\_{t+1} + \gamma (R\_{t+2} + \gamma R\_{t+3} + \gamma^2 R\_{t+4} + ...)|S\_t = s] \newline
  & = & \mathbb{E}\_\pi [R\_{t+1} + \gamma R\_{t+1}|S\_t = s] \newline
  V\_\pi(s) & = & \mathbb{E}\_\pi [R\_{t+1} + \gamma V(S\_{t+1})|S\_t = s]
\end{array}

Similarly, we can write the Bellman equations for the Q function as:

\begin{array}{lcl}
  Q\_\pi(s,a) & = & \mathbb{E}\_\pi [R\_{t+1} + \gamma V(S\_{t+1})|S\_t = s, A\_t = a] \newline
  Q\_\pi(s,a) & = & \mathbb{E}\_\pi [R\_{t+1} + \gamma \mathbb{E\_{a \sim \pi}} Q(S\_{t+1},a)|S\_t = s, A\_t = a]
\end{array}


So now, we can use these value functions (Bellman equations) to get state-value and action-value at a timestep. Let's say the agent is in state $s$ and decides to take an action $a$ and arrives in *next state* $s'$ and receives reward $r$. Then, it takes an action $a'$ to go to the next state $s''$ and so on.

<figure align="center">
  <img width="500" height="300" src="/assets/img/basics_rl/transition.png" alt="Q-learning update step">
    <figcaption> Update step </figcaption>
</figure>

But we need to do that recursively and update them on the go. To generalize this update step, we use $\pi(a \vert s)$ (policy gives us the probability of taking an action in a state) with our value function like this:

\begin{array}{lcl}
  V\_\pi(s) = \sum\_{a} \pi(a|s) Q\_{\pi}(s,a) \newline
  Q\_\pi(s,a) = R(s,a) + \gamma \sum\_{s'} P\_{ss'}^{a}, V\_\pi(s') \newline
  V\_\pi(s) = \sum\_{a} \pi(a|s) (R(s,a) + \gamma \sum\_{s'} P\_{ss'}^{a}, V\_\pi(s')) \newline
  Q\_\pi(s,a) = R(s,a) + \gamma \sum\_{s'} P\_{ss'}^{a} \sum\_{a'} \pi(a'|s') Q_{\pi}(s',a')
\end{array}

The updated value functions give us the final pieces of the RL puzzle by actually updating the policy as the agent moves through the environment. But that was a lot of stuff and a lot of technical terms so let's recap.

<p><strong>Recap</strong></p>

The agent is deployed in an environment that has some **states $s$ and some actions $a$**. While the agent doesn't have direct control over which state it moves to, it does have control over the actions which when taken results in the transition to a new state $s'$.

<figure align="center">
  <img width="650" height="300" src="/assets/img/basics_rl/rl_schematic.png" alt="Reinforcement Learning Schematic">
    <figcaption>Schematic representation of an RL agent. Source: Sutton and Barto</figcaption>
</figure>


Whenever the agent makes a move, it is rewarded with either positive or negative reinforcement called a **reward $r$**. The goal then becomes to maximize this reward by balancing immediate rewards that result in short-term positive rewards and future rewards that result in long-term positive rewards. The **discount factor $\gamma$** balances these two factors. To actually make a move and know how to make a move, the agent needs some instructions and guidance which is provided by the policy. It provides a probabilistic approach to maximizing rewards. Every time the agent enters a new state, the value function (state-value and action-value) is calculated to update the policy which then becomes the basis for the successive step. This is demonstrated by the figure above: the agent is in state $s_t$ and takes action $a_t$ at time $t$ which results in the transition to state $s_{t+1}$ and a reward $r_{t+1}$. Using this reward, the value function is updated in the policy and then repeats this process till the end goal (if any) is achieved.

The field of Reinforcement Learning is huge and this post has a lot of seemingly intimidating concepts especially for a beginner. But at the end of the day, it actually just boils down to some clever math and having these basics is super important and I can vouch for that. I've found that going over some other blog posts offering differently worded explanations always helps and I'll attempt to explain more as we go. In the next blog post, we'll explore [OpenAI Gym](https://gym.openai.com), a popular platform that allows for easy implementation of RL algorithms. See you there!

Cover Image credit: Big Hero 6 via Walt Disney Pictures
