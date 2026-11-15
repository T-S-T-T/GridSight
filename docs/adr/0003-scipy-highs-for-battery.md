# 0003 SciPy's HiGHS solver for the battery LP (instead of cvxpy)

- Status: accepted (changed from the design doc)

## Context
The battery problem is a small linear program solved about 17,500 times per year of backtest.

## Decision
Build the LP matrices directly and solve with `scipy.optimize.linprog(method="highs")`.

## Consequences
One less dependency and fast solves. The formulation adds a terminal value for stored energy
(mean forecast price x discharge efficiency): without it the optimiser empties the battery at the
end of every horizon. A test pins this behaviour down.
