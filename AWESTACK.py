import  pyomo.environ as pe
import pyomo.dae as dae
import matplotlib.pyplot as plt
import numpy as np
Parameters = {
    # 使用浮点数吧
    'Nc': 45,  # number of cells
    'A': 1.25,  # effective area of membrane electrode, m2
    'w': 30.0,  # electrlyte cncentration wt%
    'P0': 1.0,  # atmospheric pressure, atm
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

Pelec = range(1000, 400000, 10000)
Te = range(330, 370, 1)

# def pid_control(dxdt,)
def AWESTACK(Para, ParaF):
    '''

    :param Para:
    :param ParaF:
    :return:
    '''

    md= pe.ConcreteModel()

    # md.t = pe.RangeSet(0,6)
    # def _init_T(md,j):
    #     return Te[j]
    # md.T = pe.Param(md.t, rule=_init_T)
    # md.t0 = pe.Var(md.t, initialize=5)
    w = Para['w']  # KOH质量分数
    md.t = dae.ContinuousSet(bounds=(0,60*12))
    md.QHeat = pe.Var(domain = pe.NonNegativeReals, initialize=0)
    def _int_T0(md):
        return md.QHeat/(0.8*(1.340*(1+w/100)+1000))+283
    md.T0 = pe.Expression(initialize=_int_T0)
    md.T = pe.Var(md.t, initialize=md.T0)
    md.T[0].fix()

    R=8.314  # J/(mol*K)
    F=96485  # C/mol

    P_Env = 1.013E5
    T_Env = 273.15+10

    Nc = Para['Nc']  # Number of Cells
    A = Para['A']  # Active area m2

    L_an = Para['L_an'] / 1000  # Electrode thickness m
    L_cat = Para['L_cat'] / 1000  # m


    dm = ParaF['dm']
    act_an0 =ParaF['act_an0']
    act_cat0 = ParaF['act_cat0']
    act_an1 = ParaF['act_an1']
    act_cat1 = ParaF['act_cat1']
    act_an2 = ParaF['act_an2']
    act_cat2 = ParaF['act_cat2']
    act_an3 = ParaF['act_an3']
    act_cat3 = ParaF['act_cat3']
    ohm_ele = ParaF['ohm_ele']


    P = 1e6 # Pressure
    md.P_c = pe.Var(md.t, initialize=P)
    md.P_a = pe.Var(md.t, initialize=P)
    def _init_J(md,j):
        if j <60*8:
            return 800
        else:
            return 2000
    md.J = pe.Expression(md.t, initialize=_init_J)
    # md.J.display()
    # def _init_voltage(b):
    #     
    # md.voltage = pe.Block(md)
    def _eq_m(md, j):
        return w * (183.1221 - 0.56845 * (md.T[j] - 273) + 984.5679 * pe.exp(
            w / 115.96277)) / 5610.5  # mole concentration

    md.m = pe.Expression(md.t, rule=_eq_m)
    def reversible_potential(b):
        def _eq_E0(b, j):
            return (1.5184 - 1.5421 * 10 ** (-3) * md.T[j] + 9.523 * 10 ** (-5) * md.T[j] * pe.log(md.T[j]) + 9.84 * 10 ** (-8) * md.T[j] ** 2)  # standard potential
        b.E0 = pe.Expression(md.t, rule=_eq_E0)

        def _eq_P_H2O_(b, j):
            return md.T[j] ** (-3.4159) * pe.exp(37.043 - 6275.7 / md.T[j])  # vapor pressure of pure water
        b.P_H2O_ = pe.Expression(md.t, rule=_eq_P_H2O_)

        def _eq_P_H2O(b, j):
            return md.T[j] ** (-3.498) * pe.exp(37.93 - 6426.32 / md.T[j]) * \
                   pe.exp(0.016214 - 0.13082 * md.m[j] + 0.1933 * pe.sqrt(
                       md.m[j]))  # pressure of the wet hydrogen and oxygen gases near the electrode
        b.P_H2O = pe.Expression(md.t, rule=_eq_P_H2O)

        def _eq_Erev(b, j):
            return b.E0[j] + R * md.T[j] / 2 / F * pe.log((P - b.P_H2O[j]) ** 1.5 * b.P_H2O_[j] / b.P_H2O[j])  # reversible voltage
        b.Erev = pe.Expression(md.t, rule=_eq_Erev)
    md.block_reversible_potential = pe.Block(rule=reversible_potential)
    def activation_overpotential(b):
        b.a_a = pe.Expression(md.t, rule=lambda b, j: 0.07832 * act_an2 + 0.001 * act_an3 * md.T[j])  # charge transfer coefficient
        b.a_c = pe.Expression(md.t, rule=lambda b, j: 0.1175 * act_cat2 + 0.00095 * act_cat3 * md.T[j])

        def _eq_theta(b, j):
            return 0.024 * (md.J[j] / 10000) ** (1 / 3)  # covering coefficient

        b.theta = pe.Expression(md.t, rule=_eq_theta)

        def _eq_Sm(b, j):
            return A * (1 - b.theta[j])

        b.Sm = pe.Expression(md.t, rule=_eq_Sm)

        def _eq_J0_an(b, j):
            return 10 ** 4 * (9 * 10 ** (-5) * (P / P_Env) ** 0.1 *pe.exp(-42000 * act_an1 / R / md.T[j] * (1 - md.T[j] / T_Env))) * act_an0  # exchange current
        b.J0_an = pe.Expression(md.t, rule=_eq_J0_an)

        def _eq_J0_cat(b, j):
            return 10 ** 4 * (1.5 * 10 ** (-4) * (P / P_Env) ** 0.1 *pe.exp(-23000 * act_cat1 / R / md.T[j] * (1 - md.T[j] / T_Env))) * act_cat0
        b.J0_cat = pe.Expression(md.t, rule=_eq_J0_cat)

        def _eq_eps(b, j):
            return  2 / 3 * b.theta[j]
        b.epsilon = pe.Expression(md.t, rule=_eq_eps)

        def _eq_Jeff(b, j):
            return md.J[j] / (1 - b.theta[j])  # effictive current
        b.Jeff = pe.Expression(md.t, rule=_eq_Jeff)

        def _eq_Eact_an(b, j):
            return R * md.T[j] / F / b.a_a[j] * pe.log(b.Jeff[j] / b.J0_an[j])
        b.Eact_an = pe.Expression(md.t, rule=_eq_Eact_an)


        def _eq_Eact_cat(b, j):
            return R * md.T[j] / F /b.a_c[j] * pe.log(b.Jeff[j] / b.J0_cat[j])
        b.Eact_cat = pe.Expression(md.t, rule=_eq_Eact_cat)

        b.Eact = pe.Expression(md.t, rule=lambda b,j:b.Eact_an[j]+b.Eact_cat[j])
    md.block_activation_overpotential = pe.Block(rule=activation_overpotential)
    def ohmic_overpotential(b):
        def _eq_sigma_a(b, j):  # electrical conductivity of Ni S/m
            return ohm_ele * (60000000 - 279650 * md.T[j] + 532 * md.T[j] ** 2 -0.38057 * md.T[j] ** 3)
        b.sigma_a = pe.Expression(md.t, rule=_eq_sigma_a)
    
        def _eq_sigma_c(b, j):
            return ohm_ele * (60000000 - 279650 * md.T[j] + 532 *md.T[j] ** 2 - 0.38057 * md.T[j] ** 3)
        b.sigma_c = pe.Expression(md.t, rule=_eq_sigma_c)
    
        def _eq_sigma_KOH(b, j):
            return 100 * (-2.041 * md.m[j] - 0.0028 * md.m[j] ** 2 +
                                             0.005332 * md.m[j] * md.T[j] + 207.2 * md.m[j] / md.T[j] +
                                             0.001043 * md.m[j] ** 3 - 0.0000003 * md.m[j] ** 2 * md.T[j] ** 2)
        b.sigma_KOH = pe.Expression(md.t, rule=_eq_sigma_KOH)
    
        def _eq_Rcat(b, j):
            return 1/ b.sigma_c[j] * L_cat / A
        b.Rcat = pe.Expression(md.t, rule=_eq_Rcat)
    
        def _eq_Ran(b, j):
            return 1 /b.sigma_a[j] * L_an / A
        b.Ran = pe.Expression(md.t, rule=_eq_Ran)
    
        def _eq_RKOH_b(b, j):
            return 1 / (1 - md.block_activation_overpotential.epsilon[j]) ** 1.5 * ((dm / 1000) * 2) / A/b.sigma_KOH[j]
        b.RKOH_b = pe.Expression(md.t, rule=_eq_RKOH_b)
    
        def _eq_Rmem(b, j):
            return (0.06 + 80 * pe.exp(-md.T[j] / 50)) / 10000/(md.block_activation_overpotential.Sm[j])
        b.Rmem = pe.Expression(md.t, rule=_eq_Rmem)
    
        def _eq_Eohm(b, j):
            return md.J[j] * A * (b.Rmem[j] + b.RKOH_b[j] + b.Ran[j] + b.Rcat[j])
        b.Eohm = pe.Expression(md.t, rule=_eq_Eohm)
    md.block_ohmic_overpotential = pe.Block(rule=ohmic_overpotential)
    f1 = 2.50
    f2 = 0.96
    def _eq_etaF(md, j):
        return md.J[j] ** 2 * f2/(f1 + md.J[j] ** 2)
    md.etaF = pe.Expression(md.t, rule=_eq_etaF)

    def _eq_E(md, j):
        return Nc * (md.block_reversible_potential.Erev[j] + md.block_ohmic_overpotential.Eohm[j] + md.block_activation_overpotential.Eact[j])
    md.E = pe.Expression(md.t, rule=_eq_E)

    def _eq_I(md, j):
        return md.J[j] * A
    md.I = pe.Expression(md.t, rule=_eq_I)


    md.Pele = pe.Var(md.t, initialize=400000)#400kW
    def _eq_Pele(md, j):
        return  md.Pele[j] == md.E[j] * md.I[j]
    md.eq_Pele = pe.Constraint(md.t, rule=_eq_Pele)

    delta_H = 285.84
    delta_S = 163.15
    md.delta_G = pe.Expression(md.t, rule=lambda md, j: delta_H - md.T[j] * delta_S / 1000)  # kJ/mol
    AStack = 15  # m2
    hStack = 30  # heat transfer coefficient
    sigma = 1.380649 * 10 ** (-23)  # Stefan-Boltzmann constant
    epsilonHeat = 0.025  # emissivity

    md.nH2prod = pe.Expression(md.t, rule=lambda md, j: md.etaF[j] * Nc * md.I[j] / 2 / F)
    def _eq_WH2(md, j):
        return md.nH2prod[j] * md.delta_G[j]
    md.WH2 = pe.Expression(md.t, rule=_eq_WH2)
    def _eq_H2(md):
        n = 0
        for m in range(len(md.t)):
            i = m+1
            if i == len(md.t):
                pass
            else:
                ti = md.t.at(i)
                ti1 = md.t.at(i+1)
                N = (md.WH2[ti] + md.WH2[ti1])*(ti1-ti)/2
                n = n+N
        return n
    md.Total = pe.Expression(rule=_eq_H2)

    def _eq_Qloss(md, j):
        return (hStack * AStack * (md.T[j] - T_Env) + sigma * AStack * epsilonHeat * md.T[j] ** 4 / 10000)/1000
    md.Qloss = pe.Expression(md.t, rule=_eq_Qloss)

    # md.VKOH = pe.Var(md.t, initialize=1, bounds=(0,1))#kg/s
    md.VKOH = pe.Param(md.t, default=0)

    def _eq_CpKOH(md, j):
        return 4236+1.075*pe.log((md.T[j]-273)/100)+(-4831+8*w/100+8*(md.T[j]-273))*w/100
    md.CpKOH = pe.Expression(md.t, rule=_eq_CpKOH)#J/kg/K


    def _eq_dQKOH(md,j):
        return md.VKOH[j]*(md.T[j]-T_Env)*md.CpKOH[j]/1000
    md.dQKOH = pe.Expression(md.t, rule=_eq_dQKOH)

    md.dH2O = pe.Expression(md.t, rule=lambda md, j:md.nH2prod[j]*18)#g/s
    def _eq_dQGas(md,j):
        return md.nH2prod[j]/2*14.11/1000*(md.T[j]-T_Env)+md.nH2prod[j]/2/32*0.9169/1000*(md.T[j]-T_Env)
    md.dQGas = pe.Expression(md.t, rule=_eq_dQGas)

    def _eq_Qtem(md, j):
        return md.Pele[j]/1000 -md.Qloss[j] - md.WH2[j]-md.dQGas[j]-md.dQKOH[j]
    md.Qtem = pe.Expression(md.t, rule=_eq_Qtem)
    m = 1340*(1+w/100)+10000
    def _init_Cstack(md,j):
        return (md.CpKOH[j]*1.340*(1+w/100)+0.46*10000)/m
    md.Cstack = pe.Expression(md.t, rule=_init_Cstack)#kj/kg/K
    md.dTdt = dae.DerivativeVar(md.T, wrt=md.t)

    def _eq_ode(md, j):
        return md.dTdt[j] == 60*md.Qtem[j] / (md.Cstack[j]/2 * m)
    md.eq_ode = pe.Constraint(md.t, rule=_eq_ode)

    # md.tsim, md.profiles = dae.Simulator(md, package='scipy').simulate(numpoints=1000)
    # discretizer = pe.TransformationFactory('dae.finite_difference')
    pe.TransformationFactory('dae.collocation').apply_to(md, nfe=12, ncp=10, scheme='LAGRANGE-RADAU')#RADAU, LEGENDRE
    # pe.TransformationFactory('dae.finite_difference').apply_to(md, nfe=100, scheme='BACKWARD')
    def _obj(md):
        return -(md.Total-md.QHeat)
    md.obj = pe.Objective(rule=_obj)
    #hydrogen-liquid separation
    # md.H2Separation_mass_balance

    solver = pe.SolverFactory('ipopt',tee=False)

    flag = solver.solve(md, tee=True, symbolic_solver_labels=True)
    md.Cstack.display()
    # md.J.pprint()
    # print(flag)
    # md.T.display()
    # print(pe.value(sum(md.nH2prod[i]*(md.t[i+1]-md.t[i]) for i in md.t)))
    # md.E.display()
    Ed_ =[]
    Id_ =[]
    Td_ = []
    nd_=[]
    t = []
    for i in md.t:
        Ed = pe.value(md.E[i])
        Id = pe.value(md.I[i])
        Td = pe.value(md.T[i])
        nd = pe.value(md.nH2prod[i])
        Ed_.append(Ed)
        Id_.append(Id)
        Td_.append(Td)
        nd_.append(nd)
        t.append(pe.value(i))

    # plt.plot(md.t, nd_)
    # plt.xlabel('Time(min)', fontsize='large')
    # plt.ylabel('Hydrogen Generation Rate(mol/s)', fontsize='large')
    # plt.show()
    plt.plot(md.t, Ed_)
    plt.xlabel('Time(min)', fontsize='large')
    plt.ylabel('Stack Voltage(V)', fontsize='large')
    plt.show()
    plt.plot(md.t, Id_)
    plt.xlabel('Time(min)', fontsize='large')
    plt.ylabel('Stack Current(A)', fontsize='large')
    plt.show()
    plt.plot(md.t, Td_)
    plt.xlabel('Time(min)', fontsize='large')
    plt.ylabel('Temperature(K)', fontsize='large')
    plt.show()
    return Ed_,Id_, Td_, nd_,t
CELL = AWESTACK(Parameters, ParametersFitted)
# md.nH2prod.display()
# md.I.display()

Ecell = CELL[0]
Icell = CELL[1]
Tcell = CELL[2]
ncell = CELL[3]
t = CELL[4]

Ed_ =[]
Id_ =[]
Td_ = []
nd_=[]

