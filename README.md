Resources that may be helpful for you:
[David Silver's lectures](https://www.davidsilver.uk/teaching/)
[Proximal Policy Optimization](https://arxiv.org/pdf/1707.06347.pdf).
[Implementation Matters in Deep Policy Gradients](https://openreview.net/pdf?id=r1etN1rtPB)
[Stable Baselines](https://github.com/hill-a/stable-baselines)

In terminal, run
    tensorboard --logdir=./runs/
to track your agent's progress

Todo:
    - Upgrade and adjust PPO to solve all classic control and Box2D [OpenAI Gym](https://gym.openai.com) environments.
    - Solve Atari RAM environments.
    - Solve Atari vision environments.
    - Explain in detail how I arrived at all hyperparameter/implementation tweak decisions.
