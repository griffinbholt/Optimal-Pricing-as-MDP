import numpy as np 
import cvxpy as cp 

## DATA

n = 2000

np.random.seed(42)
tau = np.random.normal(loc=5.39, scale=1.0, size=n)

p = cp.Variable(1)
x = cp.Variable(n)

objective = cp.Maximize(cp.log(p) + cp.log(cp.sum(x)))
constraints = [0 <= x, x <= 1, p >= 0, 
               cp.log(p) + cp.log(x) <= cp.log(tau)] # Not CONVEX
problem = cp.Problem(objective, constraints)
problem.solve()

print(problem.value)
print(p.value)
print(x.value)