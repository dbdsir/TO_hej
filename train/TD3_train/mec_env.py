import math
import numpy as np
from typing import TypeVar, Tuple
from operator import itemgetter
from random_info import generate_random_info
from normalization import Normalization

ActType = TypeVar('ActType')
ObsType = TypeVar('ObsType')

class Mec_Env:
    action_dim = 4
    state_dim = 14
    action_bound = [-1, 1]
    time_process = 10
    time_son_process = 1
    time_node_num = int(time_process / time_son_process)
    gnb_radius = 100
    H = 2
    channel_gain_r = -30
    channel_gain_r1 = 5
    xindao = 2
    bandwidth_m = 10e9
    bandwidth_s = 20e9
    noise_t = 1e-3
    interference = 1e6
    thea = 2
    noise_s = 60
    dis = 3.6
    pass_loss_ratio = 0.02
    con_discount_factor = 0.001
    gnb_con_discount_factor = 0.0001
    s_normal = Normalization()

    def __init__(self):
        self.VEC_info = None
        self.cc_info = None
        self.task_info = None
        self.ve_info = None
        self.gnb_info = None
        self.decision_time_std = 0.01
        self.cc_time_std = 0.1
        self.weights_d = 0.25
        self.weights_con = 0.25
        self.weights_cost = 0.25
        self.weights_l = 0.25
        self.action_space = [0, 1, 2, 3, 4]
        self.T = 10
        self.M = 3
        self.C = 1
        self.T_type = 4
        self.Node_num = 1 + self.M + self.C
        self.time_node_num = 60
        self.rectangle_length = 600
        self.states = None
        self.state = None

        self.if_discrete = False

    def reset(self) -> np.ndarray:
        self.reset_env()
        self.gnb_info = self.VEC_info.generate_gnb()
        self.ve_info = self.VEC_info.generate_ve()
        self.task_info = self.VEC_info.generate_task()
        self.cc_info = self.VEC_info.generate_cc()
        self.state = np.round(
            np.append(itemgetter(*['gnb_frequency_cal', 'gnb_load_rate'])(self.gnb_info),
                      itemgetter(*['cc_frequency_cal'])(self.cc_info)),
            2)
        self.state = np.append(self.state,
                               itemgetter(*['ve_remain_cal', 've_power_cal', 've_xposition', 've_yposition'])(
                                   self.ve_info))
        self.state = np.append(self.state, itemgetter(*['task_in', 'task_cal', 'task_delay_max', ])(
            self.task_info))
        return self.state

    def reset_env(self):
        self.T = 10
        self.M = 3
        self.C = 1

        self.Node_num = 1 + self.M + self.C
        self.time_node_num = 60
        self.states = None
        self.state = None
        self.reset_info()

    def reset_info(self):
        self.VEC_info = generate_random_info(self)

    def update_env(self):
        self.ve_info = self.VEC_info.generate_ve()
        self.task_info = self.VEC_info.generate_task()
        for i in range(self.M):
            self.gnb_info['gnb_load_rate'][i] += np.random.choice([-1, 1]) * np.random.normal(2, 1)
            self.gnb_info['gnb_load_rate'][i] = np.clip(self.gnb_info['gnb_load_rate'][i], *[20, 95])
        self.state = np.round(
            np.append(itemgetter(*['gnb_frequency_cal', 'gnb_load_rate'])(self.gnb_info),
                      itemgetter(*['cc_frequency_cal'])(self.cc_info)),
            2)
        self.state = np.append(self.state,
                               itemgetter(*['ve_remain_cal', 've_power_cal', 've_xposition', 've_yposition'])(
                                   self.ve_info))
        self.state = np.append(self.state, itemgetter(*['task_in', 'task_cal', 'task_delay_max', ])(
            self.task_info))
        return self.state

    def step(self, action: np.ndarray) -> (np.ndarray, float, bool, dict):
        state, reward, done, info_dict = self.step_real(action)

        state = state.reshape(self.state_dim)
        return state, float(reward), done, info_dict

    def step_real(self, action: ActType) -> (ObsType, float, bool, dict):
        done = False
        info_dict = None
        action = (action + 1) / 2
        sign_time_out = False

        node_id, off_ratio, allo_cal_ratio, signal_launch_p = self.parsing_action(action)
        info_dict = {'node_id': node_id, 'off_ratio': off_ratio, 'allo_cal_ratio': allo_cal_ratio,
                     'signal_launch_p': signal_launch_p}
        info_dict["action"] = action
        load_value = 0
        if node_id == 0:
            delay = self.task_info['task_cal'] / (self.ve_info['ve_remain_cal'] * 10)
            consume = delay * self.ve_info['ve_power_cal']
            cost = 0
            load = 0
            delay_reward = abs(self.task_info['task_delay_max'] - delay - 0.3)
            delay_n, consume_n, cost_n, load_n = self.s_normal.reward_normal(delay_reward, consume, cost, load)
            reward = -(
                    self.weights_d * delay_n + self.weights_con * consume_n + self.weights_cost * cost_n + self.weights_l * load_n)
            if delay > self.task_info['task_delay_max']:
                reward = -20
                sign_time_out = True
            delay_1, consume_1, cost_1, load_value_1 = self.s_normal.reward_normal(delay, consume, cost, load_value)
            reward_1 = -round(
                self.weights_d * delay_1 + self.weights_con * consume_1 + self.weights_cost * cost_1 + self.weights_l * load_value_1,
                2)
            if delay > self.task_info['task_delay_max']:
                reward = -20
                reward_1 = -20
                sign_time_out = True
        else:
            xi = self.ve_info['ve_xposition']
            yi = self.ve_info['ve_yposition']
            if xi >= 10 and xi < 200:
                smx = 100
            elif xi >= 200 and xi < 400:
                smx = 300
            else:
                smx = 500

            self.channel_gain = self.channel_gain_r / (np.sqrt((smx - xi) ** 2 + yi ** 2) + self.H ** 2)
            v_trans = self.bandwidth_m * np.log2(
                1 + self.channel_gain * signal_launch_p / (
                        self.noise_t + 5 * self.channel_gain))
            t_up = self.task_info['task_in'] * off_ratio * 1024 * 1024 * 8 / 2 / (v_trans + 1e-2)
            if node_id != 4:
                x = self.gnb_info['gnb_load_rate'][node_id - 1]
                cal = self.gnb_info['gnb_frequency_cal'][node_id - 1]
                if x > 85:
                    zhekou_cal = np.around(cal * (5 * (100 - x) / 100 + 0.1), decimals=1)   # 假设负载率对计算频率有影响
                else:
                    zhekou_cal = cal
                t_attach = abs(
                    int((node_id * self.rectangle_length / 3 - 100) - self.ve_info[
                        've_xposition']) / self.rectangle_length / 2) * np.exp(
                    1 + self.pass_loss_ratio) / 20
                t_cal_mec_real = self.task_info['task_cal'] * off_ratio * 1e8 / (
                        zhekou_cal * allo_cal_ratio * 1e9 + 1e-2)
                t_cal_loc_real = self.task_info['task_cal'] * (1 - off_ratio) * 1e8 / (
                        self.ve_info['ve_remain_cal'] * 1e9 + 1e-2)
                t_cal_mec = max(t_up + t_attach + t_cal_mec_real, t_cal_loc_real)
                delay = t_cal_mec
                consume = abs(t_up * signal_launch_p + t_cal_loc_real * self.ve_info['ve_power_cal'])
                cost = abs(zhekou_cal * math.exp(
                    1 + allo_cal_ratio) * t_cal_mec_real)
                x = self.gnb_info['gnb_load_rate'][node_id - 1]
                load_update = int(
                    np.exp(allo_cal_ratio) * 2 * np.log2(2 + abs(t_cal_mec_real)))
                self.gnb_info['gnb_load_rate'][node_id - 1] += load_update
                load_server = self.gnb_info['gnb_load_rate'][node_id - 1]
                load = 0
                if load_server <= 70:
                    load = 0.4 * self.gnb_info['gnb_load_rate'][node_id - 1] / 10 - 0.8
                elif 70 < load_server <= 100:
                    load = self.gnb_info['gnb_load_rate'][node_id - 1] / 10 - 5
                mean_value_load = np.mean(self.gnb_info['gnb_load_rate'])
                load_value = np.round(
                    math.sqrt(
                        sum([(load_value - mean_value_load) ** 2 for load_value in self.gnb_info['gnb_load_rate']])), 2)

                delay_reward = abs(self.task_info['task_delay_max'] - delay - 0.3)
                delay_n, consume_n, cost_n, load_n = self.s_normal.reward_normal(delay_reward, consume, cost, load)
                reward = -(
                        self.weights_d * delay_n + self.weights_con * consume_n + self.weights_cost * cost_n + self.weights_l * load_n)

                delay_1, consume_1, cost_1, load_value_1 = self.s_normal.reward_normal(delay, consume, cost, load_value)
                reward_1 = -round(
                    self.weights_d * delay_1 + self.weights_con * consume_1 + self.weights_cost * cost_1 + self.weights_l * load_value_1,
                    2)
                if delay > self.task_info['task_delay_max']:
                    reward = -20
                    reward_1 = -20
                    sign_time_out = True
                if self.gnb_info['gnb_load_rate'][node_id - 1] >= 100 and sign_time_out == False:
                        reward = -10
                        self.gnb_info['gnb_load_rate'][node_id - 1] = 50
            else:
                s_trans = self.bandwidth_s * np.log2(
                    1 + self.channel_gain_r1 * signal_launch_p * self.dis ** (-self.thea) / (
                        self.noise_s))
                t_up_s = self.task_info['task_in'] * off_ratio * 1024 * 1024 * 8 / (s_trans + 1e-3)
                t_cal_sc = self.task_info['task_cal'] * off_ratio * 1e8 / (
                        self.cc_info['cc_frequency_cal'][0] * 1e9)
                t_cal_loc = self.task_info['task_cal'] * (1 - off_ratio) * 1e8 / (
                        self.ve_info['ve_remain_cal'] * 1e9 + 1e-2)
                delay_sc = t_up_s + t_cal_sc + np.clip(np.random.normal(loc=0.1,
                                                                        scale=self.cc_time_std),
                                                       *[0.1, 0.1])
                delay = max(delay_sc, t_cal_loc)
                consume = abs(t_up_s * signal_launch_p + t_cal_loc * self.ve_info['ve_power_cal'])
                cost = abs(
                    10 * self.cc_info['cc_calculation_price'][0] * t_cal_sc)
                load = 0
                delay_reward = abs(self.task_info['task_delay_max'] - delay - 0.3)
                delay_n, consume_n, cost_n, load_n = self.s_normal.reward_normal(delay_reward, consume, cost, load)
                reward = -(self.weights_d * delay_n + self.weights_con * consume_n + self.weights_cost * cost_n)

                delay_1, consume_1, cost_1, load_value_1 = self.s_normal.reward_normal(delay, consume, cost, load_value)
                reward_1 = -round(
                    self.weights_d * delay_1 + self.weights_con * consume_1 + self.weights_cost * cost_1 + self.weights_l * load_value_1,
                    2)
                if delay > self.task_info['task_delay_max']:
                    reward = -20
                    reward_1 = -20
                    sign_time_out = True

        info_dict['reward', 'delay', 'consume', 'cost', 'load', "bal_load"] = self.s_normal.reward_round(reward, delay,
                                                                                                        consume,
                                                                                                        cost, load,
                                                                                                        load_value)
        info_dict['delay_n', 'consume_n', 'cost_n', 'load_n'] = delay_n, consume_n, cost_n, load_n
        info_dict[
            'reward_1', 'delay_1', 'consume_1', 'cost_1', 'bal_load_1'] = reward_1, delay_1, consume_1, cost_1, load_value_1
        info_dict['node', 'off', 'all', 'trans'] = node_id, off_ratio, allo_cal_ratio, signal_launch_p
        info_dict['sign_time_out'] = sign_time_out
        info_dict['load_value'] = load_value
        state_next = self.update_env()
        return state_next, reward, done, info_dict

    def parsing_action(self, action: ActType):
        if action[0] == 1:
            node_id = self.Node_num - 1
        else:
            node_id = int(self.Node_num * action[0])
        offloading_ratio = np.round(action[1], decimals=2)
        resource_allocation_ratio = np.round(action[2], decimals=2)
        communication_channels_allocated_ratio = np.round(0.5 + action[3] * 1.5, decimals=1)
        return node_id, offloading_ratio, resource_allocation_ratio, communication_channels_allocated_ratio
