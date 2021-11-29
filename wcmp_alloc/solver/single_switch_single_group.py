import gurobipy as gp
from gurobipy import GRB

# WCMP table limit on switch.
C = 16 * 1024

try:
    # Create a new model
    m = gp.Model("single_switch_single_group")
    m.params.NonConvex = 2

    # Create variables: w_f is intended (fractional) weight, w_i is actual
    # (integral) weight.
    w_f, w_i = [100.5, 200.1, 301.0, 399.7], []
    for n in range(len(w_f)):
        w_i.append(m.addVar(vtype=GRB.INTEGER, name="w_i_" + str(n+1)))
    z = m.addVar(vtype=GRB.INTEGER, name="z")

    # Objective is quadratic.
    obj = gp.QuadExpr();
    # fastest way to construct a large objective.
    # Params are: coeffs, var1s, var2s (must be of same size).
    obj.addTerms(w_f, w_i, [z] * len(w_f))
    # Set objective
    m.setObjective(obj, GRB.MAXIMIZE)

    # Add constraint: sum(w_i) <= C
    m.addConstr(gp.quicksum(w_i) <= C, "c0")
    # Add constraint: sum(w_i) > 0 (constraint needs to be binding, vars cannot
    # be 0, so this is a workaround)
    m.addConstr(gp.quicksum(w_i) >= 0.01, "c1")
    # Add constraint: z^2 * sum(w_f^2) * sum(w_i^2) == 1
    '''
    sum_of_sq = 0
    for v in w_f:
        sum_of_sq += v * v
    c2 = gp.QuadExpr()
    c2.addTerms([1] * len(w_i), w_i, w_i)
    m.addConstr(z * z * sum_of_sq * c2 == 1, "c2")
    # Add constraint: w_i >= 0
    m.addConstr(w_i >= [0] * len(w_i), "c3")
    '''

    # Optimize model
    m.optimize()

    for v in m.getVars():
        print('%s %g' % (v.VarName, v.X))

    print('Obj: %g' % m.ObjVal)

except gp.GurobiError as e:
    print('Error code ' + str(e.errno) + ': ' + str(e))

except AttributeError:
    print('Encountered an attribute error')
