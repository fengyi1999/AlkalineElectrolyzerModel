import sys, os
sys.path.append(os.getcwd())

import prop_data
import pyomo.environ as pe
import pyomo.dae as dae
import matplotlib.pyplot as plt
import numpy as np
R = 8.314
F = 96485
'''
     ||--x--||
anode||     ||cathode
     ||     ||
'''
Parameters = {
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
        md.A = pe.Param(initialize=self.Pa['A']) # m2
        md.Nc = pe.Param(initialize=self.Pa['Nc'])
        md.P0 = pe.Param(initialize=self.Pa['P0'])
        md.T0 = pe.Param(initialize=self.Pa['T0'])
        md.w = pe.Param(initialize=self.Pa['w'])
        md.L_an = pe.Param(initialize=self.Pa['L_an']/1000)  # m
        md.L_cat = pe.Param(initialize=self.Pa['L_cat']/1000)

        md.t = dae.ContinuousSet(bounds=(0, self.t))
        IorUorP = 0  # 0: I, 1: U, 2: P
        if IorUorP == 0:
            md.I = pe.Param(md.t, initialize=lambda md, i: self.I[0, i-1])

            md.U = pe.Var(md.t, domain=pe.NonNegativeReals, initialize=lambda md, i: self.U[0, i-1])
            md.Power = pe.Expression(md.t, rule= lambda md, i: md.I[i]*md.U[i])
        # md.I.display()
        md.T = pe.Var(md.t, initialize=md.T0)
        md.T[0].fix()

        md.electrodes = pe.Set(initialize=['anode', 'cathode'])
        md.eXt = md.electrodes*md.t
        md.Pressure = pe.Var(md.eXt, initialize=md.P0)
        md.J = pe.Expression(md.t, initialize=lambda md,e,t:md.I[t]/md.A)
        md.species = pe.Set(initialize=['H2O', 'H2', 'O2', 'KOH'])
        md.sXeXt = md.species*md.eXt
        def _init_mole_fraction(md,s,e,t):  # mol/mol
            nKOH = md.w/100/prop_data.Pro_Cons['KOH']['MW']*1000
            nH2O = (1-md.w/100)/prop_data.Pro_Cons['H2O']['MW']*1000
            if s == 'KOH':
                return nKOH/(nKOH+nH2O)  # mole concentration
            if s == 'H2O':
                return nH2O/(nKOH+nH2O)
            else:
                return 0
        md.x0 = pe.Param(md.sXeXt, rule=_init_mole_fraction)

        def _cal_Erev(md,t):
            E0 = (1.5184 - 1.5421 * 10 ** (-3) * md.T[t] + 9.523 * 10 ** (-5) * md.T[t] * pe.log(md.T[t]) +\
                  9.84 * 10 ** (-8) * md.T[t] ** 2)  # standard potential
            P_H2O_ = md.T[t] ** (-3.4159) * pe.exp(37.043 - 6275.7 / md.T[t])  # vapor pressure of pure water
            P_H2O = md.T[t] ** (-3.498) * pe.exp(37.93 - 6426.32 / md.T[t]) * \
                   pe.exp(0.016214 - 0.13082 * md.x0['KOH', 'a', t] + 0.1933 * pe.sqrt(
                       md.x0['KOH', 'a', t]))  # pressure of the wet hydrogen and oxygen gases near the electrode
            Erev = E0 + R * md.T[t] / 2 / F * pe.log((md.Pressure['a', t] - P_H2O) ** 1.5 * P_H2O_ / P_H2O)  # reversible voltage
            return Erev
        md.Erev = pe.Expression(md.t, rule=_cal_Erev)

        def _cal_theta(md,e,t):
            if e == 'a':
                return 0.048 * (md.J[t]/10000) ** (1/3)
            if e == 'c':
                return 0.024 * (md.J[t]/10000) ** (1/3)
        # pe.TransformationFactory('dae.collocation').apply_to(md, nfe=10, ncp=self.number_of_steps/10, scheme='LAGRANGE-RADAU')  # RADAU, LEGENDRE

if __name__ == '__main__':
    DynamicModel(Parameters, ParametersFitted)