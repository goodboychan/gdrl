# 그로킹 심층강화학습 

저자: 미구엘 모랄레스(Miguel Morales)
역자: 강찬석 (Chanseok Kang)

(Work In Progress - inline comment 번역 진행)

**참고:** 현재는, [docker](https://github.com/docker/docker-ce) 컨테이너에서 코드를 실행하는 것만 지원합니다. docker는 모든 시스템에서 수행할 수 있게끔 단일 환경을 만들 수 있도록 해줍니다. 기본적으로 docker를 제외한 실험에 필요한 패키지들을 미리 설치하고, 구성했기 때문에, 실험 환경에서 그냥 코드만 돌리면 됩니다.

docker를 설치하기 위해서는 구글에서 "installing docker on \<your os here>" (혹은 "<현재 os>에서 docker 설치하기")라고 검색해보기 바랍니다. GPU 환경에서 코드를 실행하기 위해서는 [nvidia-docker](https://github.com/NVIDIA/nvidia-docker)를 추가적으로 설치해야 합니다. NVIDIA docker는 docker container 내에서 호스트 PC의 GPU를 사용할 수 있게 도와줍니다. docker를 설치했다면 (GPU를 사용할 경우 nvidia-docker), 아래의 세 단계를 수행하기 바랍니다.

## 코드 수행

  1. 아래의 repository에서 코드를 가져옵니다.:  
  `git clone --depth 1 https://github.com/goodboychan/gdrl.git && cd gdrl`
  2. `gdrl` docker image를 가져옵니다:  
  `docker pull goodboychan/gdrl:v0.14`
  3. container를 수행합니다.:
     - 리눅스나 Mac 환경:  
     `docker run -it --rm -p 8888:8888 -v "$PWD"/notebooks/:/mnt/notebooks/ goodboychan/gdrl:v0.14` 
     - 윈도우 환경:  
     `docker run -it --rm -p 8888:8888 -v %CD%/notebooks/:/mnt/notebooks/ goodboychan/gdrl:v0.14`
     - 참고: GPU를 사용할 경우 `docker`대신 `nvidia-docker` 를 사용합니다.
  4. 웹브라우져를 열고 터미널에 출력된 URL에 접속합니다. (보통 이렇게 되어 있습니다.: http://localhost:8888). 비밀번호는 다음과 같습니다: `gdrl`

## 이책에 대하여

### 책의 홈페이지

원서: https://www.manning.com/books/grokking-deep-reinforcement-learning
역서: TBD

### 목차

  1. [심층 강화학습의 기초](#1-심층-강화학습의-기초)
  2. [강화학습의 수학적 기초](#2-강화학습의-수학적-기초)
  3. [순간 목표와 장기 목표 간의 균형](#3-순간-목표와-장기-목표-간의-균형)
  4. [정보의 수집과 사용 간의 균형](#4-정보의-수집과-사용-간의-균형)
  5. [에이전트의 행동 평가](#5-에이전트의-행동-평가)
  6. [에이전트의 행동 개선](#6-에이전트의-행동-개선)
  7. [더 효휼적인 목표 도달](#7-더-효휼적인-목표-도달)
  8. [가치 기반 심층 강화학습의 기초](#8-가치-기반-심층-강화학습의-기초)
  9. [더 안정적인 가치 기반 방법들](#9-더-안정적인-가치-기반-방법들)
  10. [샘플 효율적인 가치 기반 방법들](#10-샘플-효율적인-가치-기반-방법들)
  11. [정책 기울기 방법과 액터-크리틱 방법들](#11-정책-기울기-방법과-액터-크리틱-방법들)
  12. [발전된 액터-크리틱 방법들](#12-발전된-액터-크리틱-방법들)
  13. [범용 인공 지능을 위한 방법들](#13-범용-인공-지능을-위한-방법들)

### 자세한 목차

#### 1. 심층 강화학습의 기초

- Notebook 없음

#### 2. 강화학습의 수학적 기초

- \([Notebook](/notebooks/chapter_02/chapter-02.ipynb)\)
  - 몇몇 MDP에 대한 구현:
    - Bandit Walk
    - Bandit Slippery Walk (BSW)
    - Slippery Walk Three
    - Random Walk
    - Russell and Norvig's Gridworld from AIMA
    - FrozenLake
    - FrozenLake8x8

#### 3. 순간 목표와 장기 목표 간의 균형

- \([Notebook](/notebooks/chapter_03/chapter-03.ipynb)\)
  - 이상적 정책을 찾기 위한 방법 구현:
    - 정책 평가법
    - 정책 개선법
    - 정책 순환법
    - 가치 순환법

#### 4. 정보의 수집과 사용 간의 균형

- \([Notebook](/notebooks/chapter_04/chapter-04.ipynb)\)
  - 밴딧 문제를 해결하기 위한 탐색 전략 구현:
    - Random
    - Greedy
    - E-greedy ($\epsilon$-그리디)
    - E-greedy with linearly decaying epsilon (선형적으로 감가하는 $\epsilon$-그리디)
    - E-greedy with exponentially decaying epsilon (기하급수적으로 감가하는 $\epsilon$-그리디)
    - Optimistic initialization (낙관적 초기화)
    - SoftMax 
    - Upper Confidence Bound (UCB)
    - Bayesian
  
#### 5. 에이전트의 행동 평가

- \([Notebook](/notebooks/chapter_05/chapter-05.ipynb)\)
  - Implementation of algorithms that solve the prediction problem (policy estimation):
    - On-policy first-visit Monte-Carlo prediction
    - On-policy every-visit Monte-Carlo prediction
    - Temporal-Difference prediction (TD)
    - n-step Temporal-Difference prediction (n-step TD)
    - TD(λ)

#### 6. 에이전트의 행동 개선

- \([Notebook](/notebooks/chapter_06/chapter-06.ipynb)\)
  - Implementation of algorithms that solve the control problem (policy improvement):
    - On-policy first-visit Monte-Carlo control
    - On-policy every-visit Monte-Carlo control
    - On-policy TD control: SARSA
    - Off-policy TD control: Q-Learning
    - Double Q-Learning

#### 7. 더 효휼적인 목표 도달

- \([Notebook](/notebooks/chapter_07/chapter-07.ipynb)\)
  - Implementation of more effective and efficient reinforcement learning algorithms:
    - SARSA(λ) with replacing traces
    - SARSA(λ) with accumulating traces
    - Q(λ) with replacing traces
    - Q(λ) with accumulating traces
    - Dyna-Q
    - Trajectory Sampling

#### 8. 가치 기반 심층 강화학습의 기초

- \([Notebook](/notebooks/chapter_08/chapter-08.ipynb)\)
  - Implementation of a value-based deep reinforcement learning baseline:
    - Neural Fitted Q-iteration (NFQ)

#### 9. 더 안정적인 가치 기반 방법들

- \([Notebook](/notebooks/chapter_09/chapter-09.ipynb)\)
  - Implementation of "classic" value-based deep reinforcement learning methods:
    - Deep Q-Networks (DQN)
    - Double Deep Q-Networks (DDQN)

#### 10. 샘플 효율적인 가치 기반 방법들

- \([Notebook](/notebooks/chapter_10/chapter-10.ipynb)\)
  - Implementation of main improvements for value-based deep reinforcement learning methods:
    - Dueling Deep Q-Networks (Dueling DQN)
    - Prioritized Experience Replay (PER)

#### 11. 정책 기울기 방법과 액터-크리틱 방법들

- \([Notebook](/notebooks/chapter_11/chapter-11.ipynb)\)
  - Implementation of classic policy-based and actor-critic deep reinforcement learning methods:
    - Policy Gradients without value function and Monte-Carlo returns (REINFORCE)
    - Policy Gradients with value function baseline trained with Monte-Carlo returns (VPG)  
    - Asynchronous Advantage Actor-Critic (A3C)
    - Generalized Advantage Estimation (GAE)
    - \[Synchronous\] Advantage Actor-Critic (A2C)
  
#### 12. 발전된 액터-크리틱 방법들

- \([Notebook](/notebooks/chapter_12/chapter-12.ipynb)\)
  - Implementation of advanced actor-critic methods:
    - Deep Deterministic Policy Gradient (DDPG)
    - Twin Delayed Deep Deterministic Policy Gradient (TD3)
    - Soft Actor-Critic (SAC)
    - Proximal Policy Optimization (PPO)

#### 13. 범용 인공 지능을 위한 방법들

- Notebook 없음
