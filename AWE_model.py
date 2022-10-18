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
        md.w0 = pe.Param(initialize=self.Pa['w'])
        md.L_an = pe.Param(initialize=self.Pa['L_an']/1000)  # m
        md.L_cat = pe.Param(initialize=self.Pa['L_cat']/1000)
        md.dm = pe.Param(initialize=self.PaF['dm'])
        md.act_an0 = pe.Param(initialize=self.PaF['act_an0'])
        md.act_cat0 = pe.Param(initialize=self.PaF['act_cat0'])
        md.act_an1 = pe.Param(initialize=self.PaF['act_an1'])
        md.act_cat1 = pe.Param(initialize=self.PaF['act_cat1'])
        md.act_an2 = pe.Param(initialize=self.PaF['act_an2'])
        md.act_cat2 = pe.Param(initialize=self.PaF['act_cat2'])
        md.act_an3 = pe.Param(initialize=self.PaF['act_an3'])
        md.act_cat3 = pe.Param(initialize=self.PaF['act_cat3'])
        md.ohm_ele = pe.Param(initialize=self.PaF['ohm_ele'])

        md.t = dae.ContinuousSet(bounds=(0, self.t), name='time,s')
        IorUorP = 0  # 0: I, 1: U, 2: P
        if IorUorP == 0:
            md.I = pe.Param(md.t, initialize=lambda md, i: self.I[0, i-1])

            md.U = pe.Var(md.t, domain=pe.NonNegativeReals, initialize=lambda md, i: self.U[0, i-1])
            md.Power = pe.Expression(md.t, rule= lambda md, i: md.I[i]*md.U[i])
        # md.I.display()
        md.T = pe.Var(md.t, initialize=md.T0, name='temperature')
        md.T[0].fix()

        md.electrodes = pe.Set(initialize=['a', 'c'], name='anode or cathode')
        md.eXt = md.electrodes*md.t
        md.Pressure = pe.Var(md.eXt, initialize=md.P0, name='pressure, atm')
        md.J = pe.Expression(md.t, initialize=lambda md,t:md.I[t]/md.A, name='current density,A/m2')
        md.species = pe.Set(initialize=['H2O', 'H2', 'O2', 'KOH'])
        md.sXeXt = md.species*md.eXt
        md.w = pe.Var(md.eXt, initialize=md.w0, name='mass fraction, wt%')
        def _init_mole_fraction(md,s,e,t):  # mol/mol
            nKOH = md.w/100/prop_data.Pro_Cons['KOH']['MW']*1000
            nH2O = (1-md.w/100)/prop_data.Pro_Cons['H2O']['MW']*1000
            if s == 'KOH':
                return nKOH/(nKOH+nH2O)  # mole concentration
            if s == 'H2O':
                return nH2O/(nKOH+nH2O)
            else:
                return 0
        def _init_m(md,e,t):
            return md.w[e,t] * (183.1221 - 0.56845 * (md.T[t] - 273) + 984.5679 * pe.exp(
            md.w[e,t] / 115.96277)) / 5610.5  # molarity
        md.m = pe.Expression(md.eXt, rule=_init_m, name='molarity')
        md.m.display()
        def _cal_Erev(md,t):
            E0 = (1.5184 - 1.5421 * 10 ** (-3) * md.T[t] + 9.523 * 10 ** (-5) * md.T[t] * pe.log(md.T[t]) +\
                  9.84 * 10 ** (-8) * md.T[t] ** 2)  # standard potential
            P_H2O_ = md.T[t] ** (-3.4159) * pe.exp(37.043 - 6275.7 / md.T[t])  # vapor pressure of pure water
            m = (md.m['a', t]+md.m['c', t])/2
            # TODO:研究阴阳极浓度差异对可逆电位影响
            P_H2O = md.T[t] ** (-3.498) * pe.exp(37.93 - 6426.32 / md.T[t]) * \
                   pe.exp(0.016214 - 0.13082 * m + 0.1933 * pe.sqrt(m))  # pressure of the wet hydrogen and oxygen gases near the electrode
            Erev = E0 + R * md.T[t] / 2 / F * pe.log((md.Pressure['a', t] - P_H2O) ** 1.5 * P_H2O_ / P_H2O)  # reversible voltage
            return Erev
        md.Erev = pe.Expression(md.t, rule=_cal_Erev)

        def _cal_theta(md,e,t):
            if e == 'a':
                return 0.048 * (md.J[t]/10000) ** (1/3)
            if e == 'c':
                return 0.024 * (md.J[t]/10000) ** (1/3)

        md.theta = pe.Expression(md.eXt, rule=_cal_theta)
        md.Sm = pe.Expression(md.eXt, rule=lambda md, e, t: md.A*(1-md.theta[e,t]))

        def _init_a(md,e,t):
            if e == 'a':
                return 0.07832 * md.act_an2 + 0.001 * md.act_an3 * md.T[t]
            if e == 'c':
                return 0.1175 * md.act_cat2 + 0.00095 * md.act_cat3 * md.T[t]
        md.a = pe.Expression(md.eXt, rule=_init_a, name='charge transfer coefficient')

        def _cal_J0(md,e,t):
            if e == 'a':
                return 10 ** 4 * (9 * 10 ** (-5) * (md.Pressure['a',t] / 1) ** 0.1 *\
                                  pe.exp(-42000 * md.act_an1 / R / md.T[t] * (1 - md.T[t] / 298.15))) * md.act_an0
            if e == 'c':
                return 10 ** 4 * (1.5 * 10 ** (-4) * (md.Pressure['c',t] / 1) ** 0.1 *\
                                  pe.exp(-23000 * md.act_cat1 / R / md.T[t] * (1 - md.T[t] / 298.15))) * md.act_cat0
        md.J0 = pe.Expression(md.eXt, rule=_cal_J0, name='exchange current, J/m2')

        def _cal_Jeff(md,e,t):
            return md.J[t]/(1-md.theta[e,t])
        md.Jeff = pe.Expression(md.eXt, rule=_cal_Jeff)

        def _cal_Eact(md,e,t):
            return R * md.T[t] / F / md.a[e,t] * pe.log(md.Jeff[e,t] / md.J0[e,t])
        md.Eact = pe.Expression(md.eXt, rule=_cal_Eact)

        def _cal_Eohm(md,e,t):
            # TODO:假设阴阳极材料一致
            sigma_electrode = md.ohm_ele * (60000000 - 279650 * md.T[t] + 532 * md.T[t] ** 2 -0.38057 * md.T[t] ** 3)
            sigma_KOH = 100 * (-2.041 * md.m[e,t] - 0.0028 * md.m[e,t] ** 2+0.005332 * md.m[e,t] * md.T[t] +\
                               207.2 * md.m[e,t] / md.T[t]+0.001043 * md.m[e,t] ** 3 - 0.0000003 * md.m[e,t] ** 2 * md.T[t] ** 2)
            Ran = 1 /sigma_electrode * md.L_an / md.A
            Rcat = 1 /sigma_electrode * md.L_cat / md.A
            RKOH = 1 / (1 - 2/3*md.theta[e,t]) ** 1.5 * ((md.dm / 1000) * 2) / md.A / sigma_KOH
            Rmem = (0.06 + 80 * pe.exp(-md.T[t] / 50)) / 10000/(md.Sm[e,t])
            Eohm = md.J[t] * md.A * (Rmem+RKOH+Ran+Rcat)
            return Eohm
        # pe.TransformationFactory('dae.collocation').apply_to(md, nfe=10, ncp=self.number_of_steps/10, scheme='LAGRANGE-RADAU')  # RADAU, LEGENDRE
if __name__ == '__main__':
    DynamicModel(Parameters, ParametersFitted)