# -*- coding: utf-8 -*-
import numpy as np

class generate_random_info:
    def __init__(self, Mec_Env):
        self.Mec_Env = Mec_Env

    def generate_ve(self) -> dict:
        vehicle_tup = ('ve_id', 've_xposition', 've_yposition', 've_right', 've_speed', 've_remain_cal', 've_power_cal',
                       've_power_transition', 've_energy_cal')
        ve_id = np.random.randint(0, self.Mec_Env.T)
        ve_xposition = np.random.randint(10, self.Mec_Env.rectangle_length - 10)
        ve_yposition = np.random.choice([5, 8], p=[0.5, 0.5])
        ve_if_right = np.random.choice([5, 8], p=[0.5, 0.5])
        ve_speed = np.random.randint(1, 30)
        ve_remain_cal = np.round(np.random.uniform(0.2, 1.4), decimals=1)
        #ve_remain_cal = 0.2
        ve_power_cal = np.random.randint(2, 10) * 10
        ve_power_transition = np.random.randint(1, 2)
        ve_energy_cal = np.random.uniform(800, 1600)
        ve_values = [ve_id, ve_xposition, ve_yposition, ve_if_right, ve_speed, ve_remain_cal, ve_power_cal,
                     ve_power_transition, ve_energy_cal]
        vehicle_information = dict(zip(vehicle_tup, ve_values))
        return vehicle_information

    def generate_task(self) -> dict:
        ratio_cal_consume = 10
        task_tup = (
            'task_id', 'task_in', 'task_cal', 'task_delay_max', 'task_con_max')
        task_id = np.random.choice([1, 2, 3, 4], p=[12 / 25, 6 / 25, 4 / 25, 3 / 25])
        task_in = np.random.randint(1, 10) * 10
        task_cal = np.random.randint(1, 10) * task_id
        task_delay_max = 0.5 * task_id
        task_consume_max = ratio_cal_consume * task_in
        task_values = [task_id, task_in, task_cal, task_delay_max, task_consume_max]
        task_information = dict(zip(task_tup, task_values))
        return task_information

    def generate_gnb(self) -> dict:
        gnb_tup = (
            'gnb_id', 'gnb_frequency_cal', 'gnb_cal_uint_price', 'gnb_power_cal', 'gnb_load_rate', 'gnb_energy_use',
            'gnb_capacity_remain')
        gnb_id_list = np.arange(self.Mec_Env.M)
        gnb_freq_cal_list = np.around(np.random.uniform(4.2, 4.6, self.Mec_Env.M),
                                      decimals=1)
        gnb_cal_uint_price_list = gnb_freq_cal_list
        gnb_power_cal_list = np.random.choice([2000, 3000, 4000], self.Mec_Env.M, True)
        gnb_load_rate_list = np.random.randint(20, 80, self.Mec_Env.M)
        gnb_energy_use_rate_list = np.zeros(self.Mec_Env.M)
        gnb_cap_remain_list = np.random.uniform(100, 1600, self.Mec_Env.M)
        gnb_values = [gnb_id_list, gnb_freq_cal_list, gnb_cal_uint_price_list, gnb_power_cal_list, gnb_load_rate_list,
                      gnb_energy_use_rate_list,
                      gnb_cap_remain_list]
        gnb_information = dict(zip(gnb_tup, gnb_values))
        return gnb_information

    def generate_cc(self) -> dict:
        cc_tup = ('cc_id', 'cc_frequency_cal', 'cc_calculation_price')
        cc_id = 101
        cc_Freq_cal = np.round(np.random.uniform(5, 6, self.Mec_Env.C), decimals=1)
        cc_cal_unit_price = cc_Freq_cal
        cc_values = [cc_id, cc_Freq_cal, cc_cal_unit_price]
        cc_information = dict(zip(cc_tup, cc_values))
        return cc_information
