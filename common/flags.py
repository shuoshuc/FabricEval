# VERBOSE=0: no Gurobi log. No informational prints.
# VERBOSE=1: Gurobi final log only. Informational prints.
# VERBOSE=2: full Gubrobi log.
VERBOSE = 1

# True means the block total ingress should equal its total egress.
EQUAL_INGRESS_EGRESS = False

# Flag to control whether to enable hedging.
ENABLE_HEDGING = True

# Spread in (0, 1] used by the hedging constraint.
S = 0.2

# If True, feeds GroupReduction solver with scaled up integer groups.
USE_INT_INPUT_GROUPS = False

# Broadcom Tomahawk 2 ECMP table limit.
TABLE_LIMIT = 16 * 1024
