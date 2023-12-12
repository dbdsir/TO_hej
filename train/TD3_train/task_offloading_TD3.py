# -*- coding: utf-8 -*-
import os
import time
from argparse import ArgumentParser
from copy import deepcopy
import datetime
import numpy as np
import pytz
import torch
import torch.nn as nn
from torch import Tensor
from mec_env import Mec_Env
from normalization import Normalization


class Config:
    def __init__(self, agent_class=None, env_class=None, env_args=None):
        self.agent_class = agent_class
        self.if_off_policy = True
        self.env_class = env_class
        self.env_args = env_args
        if env_args is None:
            env_args = {'env_name': None, 'state_dim': None, 'action_dim': None, 'if_discrete': None}
        self.env_name = env_args['env_name']
        self.state_dim = env_args['state_dim']
        self.action_dim = env_args['action_dim']
        self.if_discrete = env_args['if_discrete']
        self.gamma = 0.99
        self.reward_scale = 1.0
        self.net_dims = (64, 32)
        self.learning_rate = 6e-5
        self.soft_update_tau = 5e-3
        self.state_value_tau = 0.1
        self.batch_size = int(256)
        self.horizon_len = int(512)
        self.buffer_size = int(1e6)
        self.repeat_times = 1.
        self.gpu_id = int(0)
        self.thread_num = int(8)
        self.random_seed = int(0)
        self.cwd = None
        self.if_remove = True
        self.break_step = +np.inf
        self.eval_times = int(5)
        self.eval_per_step = int(2e3)

    def init_before_training(self):
        beijing_tz = pytz.timezone('Asia/Shanghai')
        beijing_time = datetime.datetime.now(beijing_tz)
        if self.cwd is None:
            self.cwd = f'./{self.env_name}_{self.agent_class.__name__[5:]}_{"_".join(str(i) for i in self.net_dims)}_{"{}".format(beijing_time.date())}_{"{}".format(beijing_time.hour)}_{beijing_time.minute}'  # 则使用环境名称、代理类名和随机种子设置一个新路径。
        os.makedirs(self.cwd, exist_ok=True)


class ActorBase(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.net = None
        self.explore_noise_std = None
        self.ActionDist = torch.distributions.normal.Normal
        self.state_avg = nn.Parameter(torch.zeros((state_dim,)),
                                      requires_grad=False)
        self.state_std = nn.Parameter(torch.ones((state_dim,)), requires_grad=False)

    def state_norm(self, state: Tensor) -> Tensor:
        return (state - self.state_avg) / self.state_std  # todo state_norm


class Actor(ActorBase):
    def __init__(self, dims: [int], state_dim: int, action_dim: int):
        super().__init__(state_dim=state_dim, action_dim=action_dim)
        self.net = build_mlp(dims=[state_dim, *dims, action_dim])
        layer_init_with_orthogonal(self.net[-1], std=0.5)

    def forward(self, state: Tensor) -> Tensor:
        action = self.net(state)
        return action.tanh()

    def get_action(self, state: Tensor) -> Tensor:
        action_avg = self.net(state).tanh()
        dist = self.ActionDist(action_avg,
                               self.explore_noise_std)
        action = dist.sample()
        return action.clip(-1.0, 1.0)


class CriticBase(nn.Module):  # todo state_norm, value_norm
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.net = None

        self.state_avg = nn.Parameter(torch.zeros((state_dim,)), requires_grad=False)
        self.state_std = nn.Parameter(torch.ones((state_dim,)), requires_grad=False)
        self.value_avg = nn.Parameter(torch.zeros((1,)), requires_grad=False)
        self.value_std = nn.Parameter(torch.ones((1,)), requires_grad=False)

    def state_norm(self, state: Tensor) -> Tensor:
        return (state - self.state_avg) / self.state_std  # todo state_norm

    def value_re_norm(self, value: Tensor) -> Tensor:
        return value * self.value_std + self.value_avg  # todo value_norm


class CriticTwin(CriticBase):
    def __init__(self, dims: [int], state_dim: int, action_dim: int):
        super().__init__(state_dim=state_dim, action_dim=action_dim)
        self.enc_sa = build_mlp(dims=[state_dim + action_dim, *dims])
        self.dec_q1 = build_mlp(dims=[dims[-1], action_dim])
        self.dec_q2 = build_mlp(dims=[dims[-1], action_dim])
        layer_init_with_orthogonal(self.dec_q1[-1], std=0.5)
        layer_init_with_orthogonal(self.dec_q2[-1], std=0.5)

    def forward(self, state: Tensor, action: Tensor) -> Tensor:
        state = self.state_norm(state)
        sa_tmp = self.enc_sa(torch.cat((state, action), dim=1))
        value = self.dec_q1(sa_tmp)
        value = self.value_re_norm(value)
        return value

    def get_q1_q2(self, state, action):
        state = self.state_norm(state)
        sa_tmp = self.enc_sa(torch.cat((state, action), dim=1))
        value1 = self.value_re_norm(self.dec_q1(sa_tmp))
        value2 = self.value_re_norm(self.dec_q2(sa_tmp))
        return value1, value2


def layer_init_with_orthogonal(layer, std=1.0, bias_const=1e-6):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)


def build_mlp(dims: [int]) -> nn.Sequential:
    net_list = []
    for i in range(len(dims) - 1):
        net_list.extend([nn.Linear(dims[i], dims[i + 1]), nn.ReLU()])
    del net_list[-1]
    return nn.Sequential(*net_list)


def kwargs_filter(function, kwargs: dict) -> dict:
    import inspect
    sign = inspect.signature(function).parameters.values()
    sign = {val.name for val in sign}
    common_args = sign.intersection(kwargs.keys())
    return {key: kwargs[key] for key in common_args}


def build_env(env_class=None, env_args=None):
    env = env_class(**kwargs_filter(env_class.__init__, env_args.copy()))
    for attr_str in ('env_name', 'state_dim', 'action_dim', 'if_discrete'):
        setattr(env, attr_str, env_args[attr_str])
    return env


class AgentBase:
    def __init__(self, net_dims: [int], state_dim: int, action_dim: int, gpu_id: int = 0, args: Config = Config()):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = args.gamma
        self.batch_size = args.batch_size
        self.repeat_times = args.repeat_times
        self.reward_scale = args.reward_scale
        self.learning_rate = args.learning_rate
        self.if_off_policy = args.if_off_policy
        self.soft_update_tau = args.soft_update_tau
        self.state_value_tau = args.state_value_tau
        self.last_state = None
        self.device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")
        act_class = getattr(self, "act_class", None)
        cri_class = getattr(self, "cri_class", None)
        self.act = self.act_target = act_class(net_dims, state_dim, action_dim).to(self.device)
        self.cri = self.cri_target = cri_class(net_dims, state_dim, action_dim).to(self.device) \
            if cri_class else self.act
        self.act_optimizer = torch.optim.Adam(self.act.parameters(), self.learning_rate)
        self.cri_optimizer = torch.optim.Adam(self.cri.parameters(), self.learning_rate) \
            if cri_class else self.act_optimizer
        self.criterion = torch.nn.SmoothL1Loss()

    @staticmethod
    def optimizer_update(optimizer, objective: Tensor):
        optimizer.zero_grad()
        objective.backward()
        optimizer.step()

    @staticmethod
    def soft_update(target_net: torch.nn.Module, current_net: torch.nn.Module, tau: float):
        for tar, cur in zip(target_net.parameters(), current_net.parameters()):
            tar.data.copy_(cur.data * tau + tar.data * (1.0 - tau))

    def update_avg_std_for_state_value_norm(self, states: Tensor, returns: Tensor):
        tau = self.state_value_tau
        if tau == 0:
            return

        state_avg = states.mean(dim=0, keepdim=True)
        state_std = states.std(dim=0, keepdim=True)
        self.act.state_avg[:] = self.act.state_avg * (1 - tau) + state_avg * tau
        self.act.state_std[:] = self.cri.state_std * (1 - tau) + state_std * tau + 1e-4
        self.cri.state_avg[:] = self.act.state_avg
        self.cri.state_std[:] = self.cri.state_std

        returns_avg = returns.mean(dim=0)
        returns_std = returns.std(dim=0)
        self.cri.value_avg[:] = self.cri.value_avg * (1 - tau) + returns_avg * tau
        self.cri.value_std[:] = self.cri.value_std * (1 - tau) + returns_std * tau + 1e-4


class AgentTD3(AgentBase):
    def __init__(self, net_dims: [int], state_dim: int, action_dim: int, gpu_id: int = 0, args: Config = Config()):
        self.act_class = getattr(self, 'act_class', Actor)
        self.cri_class = getattr(self, 'cri_class', CriticTwin)
        super().__init__(net_dims, state_dim, action_dim, gpu_id, args)
        self.cri_target = deepcopy(self.cri)
        self.act_target = deepcopy(self.act)
        self.explore_noise_std = getattr(args, 'explore_noise_std', 0.06)
        self.policy_noise_std = getattr(args, 'policy_noise_std', 0.12)
        self.update_freq = getattr(args, 'update_freq', 2)
        self.horizon_len = 0

    def explore_env(self, env, horizon_len: int, if_random: bool = False) -> [Tensor]:
        self.act.explore_noise_std = self.act_target.explore_noise_std = self.explore_noise_std
        states = torch.zeros((horizon_len, self.state_dim), dtype=torch.float32).to(self.device)
        actions = torch.zeros((horizon_len, self.action_dim), dtype=torch.float32).to(self.device)
        rewards = torch.zeros(horizon_len, dtype=torch.float32).to(self.device)
        dones = torch.zeros(horizon_len, dtype=torch.bool).to(self.device)
        ary_state = env.reset()
        get_action = self.act.get_action
        s_normal = Normalization()
        for i in range(horizon_len):
            state_normal = s_normal.state_normal(ary_state)
            state = torch.as_tensor(state_normal, dtype=torch.float32, device=self.device)
            action = torch.rand(self.action_dim) * 2 - 1.0 if if_random else get_action(state.unsqueeze(0))[0]
            states[i] = state
            actions[i] = action
            ary_action = action.detach().cpu().numpy()
            ary_state, reward, done, _ = env.step(ary_action)
            if done:
                ary_state = env.reset()
            rewards[i] = reward
            dones[i] = done

        self.last_state = ary_state
        rewards = rewards.unsqueeze(1)
        undones = (1.0 - dones.type(torch.float32)).unsqueeze(1)
        return states, actions, rewards, undones

    def update_net(self, buffer) -> [float]:  # 参数更新
        self.act.explore_noise_std = self.act_target.explore_noise_std = self.policy_noise_std
        states = buffer.states[-self.horizon_len:]
        reward_sums = buffer.rewards[-self.horizon_len:] * (1 / (1 - self.gamma))
        self.update_avg_std_for_state_value_norm(
            states=states.reshape((-1, self.state_dim)),
            returns=reward_sums.reshape((-1,))
        )

        obj_critics = obj_actors = 0.0
        # update_times = int(buffer.cur_size * self.repeat_times / self.batch_size)
        update_times = 20
        for t in range(update_times):
            obj_critic, state = self.get_obj_critic(buffer, self.batch_size)
            self.optimizer_update(self.cri_optimizer, obj_critic)
            self.soft_update(self.cri_target, self.cri, self.soft_update_tau)
            obj_critics += obj_critic.item()

            if t % self.update_freq == 0:
                action = self.act(state)
                obj_actor = (self.cri(state, action)).mean()
                self.optimizer_update(self.act_optimizer, -obj_actor)
                self.soft_update(self.act_target, self.act, self.soft_update_tau)
                obj_actors += obj_actor.item()
        return obj_critics / update_times, obj_actors / (update_times / self.update_freq)

    def get_obj_critic(self, buffer, batch_size: int) -> (Tensor, Tensor):
        with torch.no_grad():
            state, action, reward, undone, next_state = buffer.sample(batch_size)

            next_action = self.act.get_action(next_state)
            next_q = torch.min(*self.cri_target.get_q1_q2(next_state, next_action))
            q_label = reward + undone * self.gamma * next_q
        q1, q2 = self.cri.get_q1_q2(state, action)
        obj_critic = (self.criterion(q1, q_label) + self.criterion(q2, q_label)) / 2.
        return obj_critic, state


class ReplayBuffer:
    def __init__(self, max_size: int, state_dim: int, action_dim: int, gpu_id: int = 0):
        self.p = 0  # pointer
        self.if_full = False
        self.cur_size = 0
        self.max_size = max_size
        self.device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")
        self.states = torch.empty((max_size, state_dim), dtype=torch.float32, device=self.device)
        self.actions = torch.empty((max_size, action_dim), dtype=torch.float32, device=self.device)
        self.rewards = torch.empty((max_size, 1), dtype=torch.float32, device=self.device)
        self.undones = torch.empty((max_size, 1), dtype=torch.float32, device=self.device)

    def update(self, items: [Tensor]):
        states, actions, rewards, undones = items
        p = self.p + rewards.shape[0]
        if p > self.max_size:
            self.if_full = True
            p0 = self.p
            p1 = self.max_size
            p2 = self.max_size - self.p
            p = p - self.max_size
            self.states[p0:p1], self.states[0:p] = states[:p2], states[-p:]
            self.actions[p0:p1], self.actions[0:p] = actions[:p2], actions[-p:]
            self.rewards[p0:p1], self.rewards[0:p] = rewards[:p2], rewards[-p:]
            self.undones[p0:p1], self.undones[0:p] = undones[:p2], undones[-p:]
        else:
            self.states[self.p:p] = states
            self.actions[self.p:p] = actions
            self.rewards[self.p:p] = rewards
            self.undones[self.p:p] = undones
        self.p = p
        self.cur_size = self.max_size if self.if_full else self.p

    def sample(self, batch_size: int) -> [Tensor]:
        ids = torch.randint(self.cur_size - 1, size=(batch_size,), requires_grad=False)
        return self.states[ids], self.actions[ids], self.rewards[ids], self.undones[ids], self.states[ids + 1]


def train_agent(args: Config):
    args.init_before_training()
    gpu_id = args.gpu_id
    env = build_env(args.env_class, args.env_args)
    agent = args.agent_class(args.net_dims, args.state_dim, args.action_dim, gpu_id=gpu_id, args=args)
    agent.last_state = env.reset()
    filename = f"{args.cwd}/train.txt"
    with open(filename, "a") as file:
        file.write("\n==============================训练参数================================\n\n")
        file.write(
            f"| {'学习率':>8}  {'折扣率':>8}  " f"| {'终止阈值':>8}  {'探索步长':>8}  " f"| {'批次大小':>8} {'网络结构':>8}\n")
        file.write(
            f"\n| {args.learning_rate:10.6f}  {args.gamma:>10.2f}  | {args.break_step:10.0f}  {args.eval_per_step:12.0f}  | {args.batch_size:8.0f} {args.net_dims:}\n\n")

    evaluator = Evaluator(eval_env=build_env(args.env_class, args.env_args),
                          eval_per_step=args.eval_per_step, eval_times=args.eval_times, cwd=args.cwd)
    buffer = ReplayBuffer(gpu_id=gpu_id, max_size=args.buffer_size,
                          state_dim=args.state_dim, action_dim=1 if args.if_discrete else args.action_dim, )
    buffer_items = agent.explore_env(env, args.horizon_len * args.eval_times, if_random=True)
    buffer.update(buffer_items)

    torch.set_grad_enabled(False)
    while True:
        buffer_items = agent.explore_env(env, args.horizon_len)
        buffer.update(buffer_items)
        torch.set_grad_enabled(True)
        logging_tuple = agent.update_net(buffer)
        torch.set_grad_enabled(False)
        evaluator.evaluate_and_save(agent.act, args.horizon_len, logging_tuple)
        if (evaluator.total_step > args.break_step) or os.path.exists(f"{args.cwd}/stop"):
            break
    evaluator.close()


class Evaluator:
    def __init__(self, eval_env, eval_per_step: int = 1e4, eval_times: int = 8, cwd: str = '.'):
        self.cwd = cwd
        self.env_eval = eval_env
        self.eval_step = 0
        self.total_step = 0
        self.start_time = time.time()
        self.eval_times = eval_times
        self.eval_per_step = eval_per_step
        self.recorder = list()
        filename = f"{self.cwd}/train.txt"
        with open(filename, "a") as file:
            file.write("==============================训练信息================================\n")
            file.write(
                f"\n| {'step':>8}  {'time':>8}  | {'avgR':>8}  {'stdR':>6}  {'avgS':>6}  | {'objC':>8}  {'objA':>8}\n")

    def evaluate_and_save(self, actor, horizon_len: int, logging_tuple: tuple):
        self.total_step += horizon_len
        if self.eval_step + self.eval_per_step > self.total_step:
            return
        self.eval_step = self.total_step
        rewards_steps_ary = [get_rewards_and_steps(self.env_eval, actor) for _ in range(self.eval_times)]
        rewards_steps_ary = np.array(rewards_steps_ary, dtype=np.float32)
        avg_r = rewards_steps_ary[:, 0].mean()
        std_r = rewards_steps_ary[:, 0].std()
        avg_s = rewards_steps_ary[:, 1].mean()
        used_time = time.time() - self.start_time
        self.recorder.append((self.total_step, used_time, avg_r))
        save_path = f"{self.cwd}/actor_{self.total_step:08.0f}_{used_time:08.0f}_{avg_r:08.2f}.pth"
        if self.total_step % 20480 == 0:
            torch.save(actor.state_dict(), save_path)
        filename = f"{self.cwd}/train.txt"
        x = f"| {self.total_step:8.2e}  {used_time:8.0f}  " f"| {avg_r:8.2f}  {std_r:6.2f}  {avg_s:6.0f}  " f"| {logging_tuple[0]:8.2f}  {logging_tuple[1]:8.2f} \n"
        with open(filename, "a") as file:
            file.write(x)
        print(x)

    def close(self):
        np.save(f"{self.cwd}/recorder.npy", np.array(self.recorder))
        information_of_train = np.load(f"{self.cwd}/recorder.npy")
        print('训练信息\n', information_of_train)
        filename = f"{self.cwd}/train.txt"
        with open(filename, "a") as file:
            np.savetxt(file, information_of_train, fmt=['%.3e', '%.2e', '%.3e'],
                       header='\nsteps   时间-time  平均rewards',
                       footer='************************************************',
                       comments=' ')
        draw_learning_curve_using_recorder(self, self.cwd)


def get_rewards_and_steps(env, actor, if_render: bool = False) -> (float, int):
    device = next(actor.parameters()).device
    state = env.reset()
    episode_steps = 0
    cumulative_returns = 0.0
    s_normal = Normalization()
    for episode_steps in range(100):
        state_normal = s_normal.state_normal(state)
        tensor_state = torch.as_tensor(state_normal, dtype=torch.float32, device=device).unsqueeze(0)
        tensor_action = actor(tensor_state)
        action = tensor_action.detach().cpu().numpy()[0]
        state, reward, done, _ = env.step(action)
        # if episode_steps % 2000 == 0:
        #     print("action_test", tensor_action, _['node_id'], _['off_ratio'], _['allo_cal_ratio'], _['signal_launch_p'],
        #           _['delay_n', 'consume_n', 'cost_n', 'load_n'], "  reward_normal:",
        #           _['reward', 'delay', 'consume', 'cost', 'load'])
        cumulative_returns += reward

        if if_render:
            env.render()
        if done:
            break
    cumulative_returns = getattr(env, 'cumulative_returns', cumulative_returns)  # todo
    return cumulative_returns, episode_steps + 1


def draw_learning_curve_using_recorder(self, cwd: str):
    recorder = np.load(f"{cwd}/recorder.npy")
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    x_axis = recorder[:, 0]
    y_axis = recorder[:, 2]
    plt.plot(x_axis, y_axis)
    plt.xlabel('#samples (Steps)')
    plt.ylabel('#Rewards (Score)')
    plt.grid()
    file_path = f"{cwd}/LearningCurve{self.start_time}.jpg"
    plt.savefig(file_path)
    print(f"| Save learning curve in {file_path}")


def evaluate_agent(args: Config):
    pass


def train_td3_for_task_offloading(args):
    env_args = {
        'env_name': 'IoV_MEC_ENV',
        'state_dim': 14,
        'action_dim': 4,
        'if_discrete': False
    }
    parameter = Config(agent_class=AgentTD3, env_class=Mec_Env, env_args=env_args)
    parameter.break_step = int(60e4)
    parameter.net_dims = (256, 256, 256)
    parameter.gpu_id = args.gpu
    parameter.gamma = 0.98
    parameter.learning_rate = args.lr
    bool_train = 1
    if bool_train:
        train_agent(parameter)
    else:
        evaluate_agent(parameter)


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Research on task offloading based on deep reinforcement learning")
    parser.add_argument("--input_size", type=int, default=14)
    parser.add_argument("--output_size", type=int, default=4)
    parser.add_argument('--T', type=int, default=10, help='number of task vehicle')
    parser.add_argument('--M', type=int, default=5, help='number of Mobile Edge Computing')
    parser.add_argument('--C', type=int, default=1, help='number of Control Base Station')
    parser.add_argument('--N', type=int, default=100, help='Number of sub channels of a single base station')
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rete')
    parser.add_argument('--gamma', type=float, default=0.98, help='discount factor')
    parser.add_argument('--epsilon', type=float, default=0.1)
    args = parser.parse_args()
    train_td3_for_task_offloading(args)
