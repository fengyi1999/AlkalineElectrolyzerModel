import pyomo.environ as pe
import pyomo.dae as dae
import matplotlib.pyplot as plt
import numpy as np

Parameters = {
    # 使用浮点数吧
    'Nc': 45,  # number of cells
    'A': 1.25,  # effective area of membrane electrode, m2
    'w': 30.0,  # electrlyte cncentration wt%
    'P0': 1.0,  # atmospheric pressure, atm
    'T0':300,
    'L_an': 2.0,  # Electrode thickness mm
    'L_cat': 2.0,  # mm
}
ParametersFitted = {
    'dm': 3.189613967230337,  # mm
    'act_an0': 4.25340646928921,
    'act_cat0': 73.10620521967346,
    'act_an1': 17.87020316392038,
    'act_cat1': 0.0006317084948595313 ,
    'act_an2': 145.87124089666054 ,
    'act_cat2': 0.0012536801589563605,
    'act_an3': 0.031428255546757655,
    'act_cat3': 0.6680042799433734,
    'ohm_ele': 100000.03491654659,
}
class DynamicModel(object):
    def __init__(self, Pa, PaF):
        self.T0 = 300  # 初始温度，K
        self.t = 12*60  # 模拟时间，s
        # self.number_of_steps = 200  # 模拟步数
        self.I = np.ones((1, 12*60))
        self.U = np.ones((1, self.t))
        self.P = np.ones((1, self.t))
        self.Pa = Pa
        self.PaF = PaF
        self.build()

    def build(self):
        md = pe.ConcreteModel()
        md.A = pe.Param(initialize=self.Pa['A'])
        md.Nc = pe.Param(initialize=self.Pa['Nc'])
        md.P0 = pe.Param(initialize=self.Pa['P0'])
        md.T0 = pe.Param(initialize=self.Pa['T0'])
        md.w = pe.Param(initialize=self.Pa['w'])
        md.L_an = pe.Param(initialize=self.Pa['L_an'])
        md.L_cat = pe.Param(initialize=self.Pa['L_cat'])

        md.t = dae.ContinuousSet(bounds=(0, self.t))
        IorUorP = 0  # 0: I, 1: U, 2: P
        if IorUorP == 0:
            md.I = pe.Param(md.t, initialize=lambda md, i: self.I[0, i-1])
            md.U = pe.Var(md.t, domain=pe.Reals, initialize=lambda md, i: self.U[0, i-1])
            md.P = pe.Expression(md.t, rule= lambda md, i: md.I[i]*md.U[i])
        md.I.display()

        # pe.TransformationFactory('dae.collocation').apply_to(md, nfe=10, ncp=self.number_of_steps/10, scheme='LAGRANGE-RADAU')  # RADAU, LEGENDRE

if __name__ == '__main__':
    DynamicModel(Parameters, ParametersFitted)