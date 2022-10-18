from fileinput import filename
import sys, os
sys.path.append(os.getcwd())
import pandas as pd
import pyomo.environ as pe

file_name = 'data/kappa-T-wt.csv'
data = pd.read_csv(file_name)
T = list(data.columns)[1:]  # C
wt = list(data['NaN'])
number_of_data = len(T)*len(wt)

md = pe.ConcreteModel()

md.T_set = pe.Set(initialize=range(len(T)))
md.wt_set = pe.Set(initialize=range(len(wt)))
md.kappa_set = md.wt_set*md.T_set
def _init_T(md,i):
    return float(T[i])+273.15
md.T = pe.Expression(md.T_set, rule=_init_T)

def _init_wt(md,i):
    return float(wt[i])
md.wt = pe.Expression(md.wt_set, rule = _init_wt)

def _init_kappa(md,i,j):
    return data.iloc[i,j+1]
md.exp_data = pe.Expression(md.kappa_set, rule=_init_kappa)

md.A = pe.Var(initialize=0)
md.B = pe.Var(initialize=0)
md.C = pe.Var(initialize=0)
md.D = pe.Var(initialize=0)
md.E = pe.Var(initialize=0)
md.F = pe.Var(initialize=0)
md.G = pe.Var(initialize=0)
md.H = pe.Var(initialize=0)
md.I = pe.Var(initialize=0)
def _cal_kappa(md,i,j):
    return md.A*md.wt[i]+md.B*md.wt[i]*md.T[j]+md.C*md.wt[i]**2+md.D*md.T[j]**2+md.E*md.T[j]+md.F*md.wt[i]*md.T[j]**2+md.G*md.wt[i]**2*md.T[j]+\
           md.H*md.wt[i]**(-1)+md.I*md.T[j]**(-1)
md.md_data = pe.Expression(md.kappa_set, rule=_cal_kappa)
def _init_error(md,i,j):
    return (md.md_data[i,j]-md.exp_data[i,j])/md.exp_data[i,j]
md.error = pe.Expression(md.kappa_set, rule=_init_error)
def _init_obj(md):
    return sum(sum((md.md_data[i,j]-md.exp_data[i,j])**2/md.exp_data[i,j] for i in md.wt_set) for j in md.T_set)
md.obj = pe.Objective(rule=_init_obj)
solver = pe.SolverFactory('ipopt')
solver.solve(md, tee='True')
md.error.display()