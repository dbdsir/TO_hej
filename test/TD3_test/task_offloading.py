import time
from copy import deepcopy
import numpy as np
import torch
import matplotlib
import torch.nn as nn
from torch import Tensor
from mec_env import Mec_Env
from argparse import ArgumentParser
from normalization import Normalization
import GA

matplotlib.use('TkAgg')
matplotlib.rc("font", family="KaiTi")
matplotlib.rcParams["axes.unicode_minus"] = False
N = 50  # 种群数目
D = 4  # 维度
T = 50  # 最大迭代次数
c1 = c2 = 1.5  # 个体学习因子与群体学习因子
w_max = 0.8  # 权重系数最大值
w_min = 0.4  # 权重系数最小值
x_max = 1
x_min = 0
v_max = 0.2
v_min = -0.2


def func(state, x):
    y = env.step(state, x)
    return y[0]


def pso_test(ary_state, steps):
    s_normal = Normalization()
    start_time = time.time()
    for i in range(steps):
        state_normal = s_normal.state_normal(ary_state)
        # 初始化种群个体
        x = np.random.rand(N, D) * (x_max - x_min) + x_min
        v = np.random.rand(N, D) * (v_max - v_min) + v_min
        p = x
        p_best = np.ones((N, 1))
        for i in range(N):
            p_best[i] = func(state_normal, x[i, :])
        g_best = 100  # 设置真的全局最优值
        gb = np.ones(T)
        x_best = np.ones(D)
        for i in range(T):
            for j in range(N):
                if p_best[j] > func(state_normal, x[j, :]):
                    p_best[j] = func(state_normal, x[j, :])
                    p[j, :] = x[j, :].copy()
                if g_best > p_best[j]:
                    g_best = p_best[j]
                    x_best = x[j, :].copy()
                w = w_max - (w_max - w_min) * i / T
                v[j, :] = w * v[j, :] + c1 * np.random.rand(1) * (p[j, :] - x[j, :]) + c2 * np.random.rand(1) * (
                        x_best - x[j, :])
                x[j, :] = x[j, :] + v[j, :]
                for ii in range(D):
                    if (v[j, ii] > v_max) or (v[j, ii] < v_min):
                        v[j, ii] = v_min + np.random.rand(1) * (v_max - v_min)
                    if (x[j, ii] > x_max) or (x[j, ii] < x_min):
                        x[j, ii] = x_min + np.random.rand(1) * (x_max - x_min)
            gb[i] = g_best
        print("值：", gb[T - 1], "动作：", x_best, "决策:", parsing_action(x_best))
        end_time = time.time()
        reback = env.step(state, x_best)
        print("奖励：", reback[0])
        info_pso = reback[2]
        print("决策耗时:%.4f秒" % (end_time - start_time))
        return (end_time - start_time), gb[T - 1], parsing_action(x_best), info_pso


def parsing_action(action):
    if action[0] == 1:
        node_id = 4
    else:
        node_id = int(5 * action[0])
    offloading_ratio = np.round(action[1], decimals=2)
    resource_allocation_ratio = np.round(action[2], decimals=2)
    communication_channels_allocated_ratio = np.round(0.5 + action[3] * 1.5, decimals=1)
    return node_id, offloading_ratio, resource_allocation_ratio, communication_channels_allocated_ratio


def td3_test(args, state, steps):
    td3_agent = args.agent_class(args.net_dims, args.state_dim, args.action_dim, gpu_id=args.gpu_id, args=args)
    actor = td3_agent.act
    model_path = f".././model/actor_00348160_00000950_-0213.36.pth"
    actor.load_state_dict(torch.load(model_path))
    if_random = False
    s_normal = Normalization()
    start_time = time.time()
    print("状态：", state)
    state_normal = s_normal.state_normal(state)
    tensor_state = torch.as_tensor(state_normal, dtype=torch.float32,
                                   device=next(actor.parameters()).device).unsqueeze(
        0)
    tensor_action = torch.rand(4) if if_random else actor(tensor_state)
    action = tensor_action.detach().cpu().numpy()[0]
    action = (action + 1) / 2
    reward, done, _ = env.step(state_normal, action)
    print("值：", reward, "变量：", action, "决策：", parsing_action(action))
    print("奖励：", env.step(state_normal, action)[0])
    end_time = time.time()
    print("决策耗时:%.4f秒\n" % (end_time - start_time))
    return end_time - start_time, reward, _


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
        self.repeat_times = 1.0
        self.gpu_id = int(0)
        self.thread_num = int(8)
        self.random_seed = int(0)
        self.cwd = None
        self.if_remove = True
        self.break_step = +np.inf
        self.eval_times = int(20)
        self.eval_per_step = int(2e3)


class ActorBase(nn.Module):  # todo state_norm
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
        return value1, value2  # two Q values


def layer_init_with_orthogonal(layer, std=1.0, bias_const=1e-6):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)


def build_mlp(dims: [int]) -> nn.Sequential:
    net_list = []
    for i in range(len(dims) - 1):
        net_list.extend([nn.Linear(dims[i], dims[i + 1]), nn.ReLU()])
    del net_list[-1]
    return nn.Sequential(*net_list)


def get_gym_env_args(env, if_print: bool) -> dict:
    env_name = env.env_name
    state_dim = env.state_dim
    action_dim = env.action_dim
    if_discrete = env.if_discrete
    env_args = {'env_name': env_name, 'state_dim': state_dim, 'action_dim': action_dim, 'if_discrete': if_discrete}
    print(f"env_args = {repr(env_args)}") if if_print else None
    return env_args


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
        self.horizon_len = 0

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

    def update_net(self, buffer) -> [float]:
        self.act.explore_noise_std = self.act_target.explore_noise_std = self.policy_noise_std
        states = buffer.states[-self.horizon_len:]
        reward_sums = buffer.rewards[-self.horizon_len:] * (1 / (1 - self.gamma))
        self.update_avg_std_for_state_value_norm(
            states=states.reshape((-1, self.state_dim)),
            returns=reward_sums.reshape((-1,))
        )

        obj_critics = obj_actors = 0.0
        update_times = int(buffer.cur_size * self.repeat_times / self.batch_size)
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

    def step(self, action: np.ndarray) -> (np.ndarray, float, bool, dict):
        state, reward, done, info_dict = self.env.step(action * 2)
        state = state.reshape(self.state_dim)
        return state, float(reward * 0.5), done, info_dict


class Evaluator:
    def __init__(self, eval_env, eval_per_step: int = 1e4, eval_times: int = 8, cwd: str = '.'):
        self.cwd = cwd
        self.env_eval = eval_env
        self.eval_step = 0
        self.total_step = 0
        self.eval_times = eval_times
        self.eval_per_step = eval_per_step
        self.recorder = list()

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


def td3_pre():
    parser = ArgumentParser(
        description="Research on task offloading based on deep reinforcement learning")
    parser.add_argument("--input_size", type=int, default=14)
    parser.add_argument("--output_size", type=int, default=4)
    parser.add_argument('--T', type=int, default=10, help='number of task vehicle')
    parser.add_argument('--M', type=int, default=5, help='number of Mobile Edge Computing')
    parser.add_argument('--C', type=int, default=1, help='number of Control Base Station')
    parser.add_argument('--N', type=int, default=100, help='Number of sub channels of a single base station')
    parser.add_argument('--time_node_num', type=int, default=40, help='Number of time nodes')
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rete')
    parser.add_argument('--gamma', type=float, default=0.98, help='discount factor')
    parser.add_argument('--epsilon', type=float, default=0.1)
    env_args = {
        'env_name': 'IoV_MEC_ENV',
        'state_dim': 14,
        'action_dim': 4,
        'if_discrete': False
    }
    args = parser.parse_args()
    parameter = Config(agent_class=AgentTD3, env_class=Mec_Env, env_args=env_args)
    parameter.break_step = int(5e4)
    parameter.net_dims = (256, 256, 256)
    parameter.gpu_id = args.gpu
    parameter.gamma = 0.98
    parameter.learning_rate = args.lr
    return parameter


def information_origin(x):
    info = [x['reward', 'delay', 'consume', 'cost', 'load', 'bal_load'][i] for i in range(0, 6)]
    info.extend([1, x['load_value']] if x['node_id'] != 0 and x['node_id'] != 4 else [0, 0])
    return info


def information_now(x):
    info_1 = [x['reward_1', 'delay_1', 'consume_1', 'cost_1', 'bal_load_1'][i] for i in range(0, 5)]
    return info_1


if __name__ == "__main__":
    steps = 100
    dec_time = [0] * 3
    reward = [0] * 3
    sys_benefits = [0] * 3
    cs = [0] * 3
    pso = [0] * 8
    ga = [0] * 8
    td3 = [0] * 8
    filename = f"{time.time()}_test.txt"
    with open(filename, "a") as file:
        file.write("\n===========================测试信息==============================")
    env = Mec_Env()
    for i in range(steps):
        state = env.reset()
        print("状态：", state)
        explore_steps = 1
        a, b, c, d = pso_test(state, explore_steps)
        dec_time[0] += a
        info_pso = information_origin(d)
        info_pso_1 = information_now(d)
        if d['sign_time_out']:
            cs[0] += 1
        for j in range(len(info_pso)):
            pso[j] += info_pso[j]
        reward[0] += info_pso[0]
        sys_benefits[0] += info_pso_1[0]
        result = GA.dd(env, state)
        dec_time[1] += result["executeTime"]
        r2 = result["ObjV"][0][0]
        action2 = result["Vars"][0]
        o, p, q = env.step(state, action2)
        print("卸载决策：", parsing_action(action2))
        info_ga = information_origin(q)
        info_ga_1 = information_now(q)
        if q['sign_time_out']:
            cs[1] += 1
        for m in range(len(info_ga)):
            ga[m] += info_ga[m]
        reward[1] += info_ga[0]
        sys_benefits[1] += info_ga_1[0]
        e, f, g = td3_test(td3_pre(), state, explore_steps)
        dec_time[2] += e
        info_td3 = information_origin(g)
        info_td3_1 = information_now(g)
        if g['sign_time_out']:
            cs[2] += 1
        for k in range(len(info_td3)):
            td3[k] += info_td3[k]
        reward[2] += info_td3[0]
        sys_benefits[2] += info_td3_1[0]

        with open(filename, "a") as file:
            file.write(f"\n第{i + 1}次决策:\n")
            file.write(f"\n状态：{state}\n")
            file.write(f"\n粒子群优化：卸载决策   ：{c}")
            file.write(
                f"\n决策耗时：{format(dec_time[0], '20.3f')},超时率：{format(cs[0] / (i + 1), '10.3f')}")
            file.write(f"\n归一化奖励_训练：{d['delay_n', 'consume_n', 'cost_n', 'load_n']}")
            file.write(f"\n归一化奖励：{d['reward_1', 'delay_1', 'consume_1', 'cost_1', 'bal_load_1']}")
            file.write(f"\n总奖励：{format(reward[0], '20.3f')},总时延：{format(pso[1] / (i + 1))}")
            file.write(f"\n总系统效益：{format(sys_benefits[0], '20.3f')}")
            file.write(f"\n总能耗：{format(pso[2] / (i + 1), '20.3f')},总花费：{format(pso[3] / (i + 1), '20.3f')}")
            file.write(f"\n总负载均衡标准差：{format(pso[7] / (pso[6] + 1e-4), '20.3f')}\n")
            file.write(f"\n遗传算法  ：卸载决策   ： {parsing_action(action2)}")
            file.write(
                f"\n耗时  ：{format(dec_time[1], '20.3f')},超时率：{format(cs[1] / (i + 1), '10.3f')}")
            file.write(f"\n归一化奖励_训练：{q['delay_n', 'consume_n', 'cost_n', 'load_n']}")
            file.write(f"\n归一化奖励：{q['reward_1', 'delay_1', 'consume_1', 'cost_1', 'bal_load_1']}")
            file.write(f"\n总奖励：{format(reward[1], '20.3f')},总时延：{format(ga[1] / (i + 1), '20.3f')}")
            file.write(f"\n总系统效益：{format(sys_benefits[1], '20.3f')}")
            file.write(f"\n总能耗：{format(ga[2] / (i + 1), '20.3f')},总花费：{format(ga[3] / (i + 1), '20.3f')}")
            file.write(f"\n总负载均衡标准差：{format(ga[7] / (ga[6] + 1e-4), '20.3f')}\n")
            file.write(f"\nTD3  ：卸载决策   ：{g['node', 'off', 'all', 'trans']}")
            file.write(
                f"\n耗时  ：{format(dec_time[2], '20.3f')},超时率：{format(cs[2] / (i + 1), '10.3f')}")
            file.write(f"\n归一化奖励_训练：{g['delay_n', 'consume_n', 'cost_n', 'load_n']}")
            file.write(f"\n归一化奖励：{g['reward_1', 'delay_1', 'consume_1', 'cost_1', 'bal_load_1']}")
            file.write(f"\n总奖励：{format(reward[2], '20.3f')},总时延：{format(td3[1] / (i + 1), '20.3f')}")
            file.write(f"\n总系统效益：{format(sys_benefits[2], '20.3f')}")
            file.write(f"\n总能耗：{format(td3[2] / (i + 1), '20.3f')},总花费：{format(td3[3] / (i + 1), '20.3f')}")
            file.write(f"\n总负载均衡标准差：{format(td3[7] / (td3[6] + 1e-4), '20.3f')}\n")
    print("\n===========================汇总结果==============================")
    print("粒子群优化:", "决策耗时：", format(dec_time[0], '.3f'), "系统效益：", format(sys_benefits[0], '.3f'))
    print("遗传算法:", "决策耗时：", format(dec_time[1], '.3f'), "系统效益：", format(sys_benefits[1], '.3f'))
    print("TD3:", "决策耗时：", format(dec_time[2], '.3f'), "系统效益：", format(sys_benefits[2], '.3f'))
