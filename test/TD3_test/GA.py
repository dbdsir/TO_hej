# -*- coding: utf-8 -*-
import matplotlib
from mec_env import Mec_Env
import numpy as np
import geatpy as ea

matplotlib.use('TkAgg')


class MyProblem(ea.Problem):

    def __init__(self, env, state):
        name = 'Mytask_offloading'
        self.env = env
        self.state = state
        M = 1
        maxormins = [1]
        Dim = 4
        varTypes = [0] * Dim
        lb = [0, 0, 0, 0]
        ub = [1, 1, 1, 1]
        lbin = [1, 0, 0, 1]
        ubin = [1, 1, 0, 1]
        ea.Problem.__init__(self,
                            name,
                            M,
                            maxormins,
                            Dim,
                            varTypes,
                            lb,
                            ub,
                            lbin,
                            ubin)

    def evalVars(self, Vars):
        x1 = Vars[:, [0]]
        x2 = Vars[:, [1]]
        x3 = Vars[:, [2]]
        x4 = Vars[:, [3]]

        x = np.concatenate((x1, x2, x3, x4), axis=1)
        f = np.zeros((50, 1))
        for i in range(len(x)):
            f[i] = self.env.step(self.state, x[i, :])[0]
        CV = np.hstack([x3 - 1])
        return f, CV

    def calReferObjV(self):
        referenceObjV = np.array([[2.5]])
        return referenceObjV


def dd(env, state):
    problem = MyProblem(env, state)
    algorithm = ea.soea_DE_currentToBest_1_bin_templet(
        problem,
        ea.Population(Encoding='RI', NIND=50),
        MAXGEN=50,
        logTras=1)
    algorithm.mutOper.F = 0.7
    algorithm.recOper.XOVR = 0.7

    prophetVars = np.array([[0.4, 0.2, 0.2, 0.4]])

    res = ea.optimize(algorithm,
                      prophet=prophetVars,
                      verbose=True,
                      drawing=0,
                      outputMsg=True,
                      drawLog=False,
                      saveFlag=True)
    return res


if __name__ == '__main__':
    env = Mec_Env()
    state = env.reset()
    dd(env, state)
