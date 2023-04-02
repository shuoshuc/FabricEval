# VERBOSE=0: no Gurobi log. No informational prints.
# VERBOSE=1: Gurobi final log only. Informational prints.
# VERBOSE=2: full Gubrobi log.
VERBOSE = 1

# Probability of a link failure in the topology. N.B., setting it too high might
# cause a network partition.
P_LINK_FAILURE = 0.0

# True means the block total ingress should equal its total egress.
EQUAL_INGRESS_EGRESS = False

# Fraction of blocks with 0 demand.
P_SPARSE = 0.15

# Flag to control whether to enable hedging.
ENABLE_HEDGING = True

# Spread in (0, 1] used by the hedging constraint.
S = 0.5

# If True, feeds GroupReduction solver with scaled up integer groups.
USE_INT_INPUT_GROUPS = False

# Infinite ECMP table size, overrides `TABLE_LIMIT` and `MAX_GROUP_SIZE` flag.
INFINITE_ECMP_TABLE = False

# Example (Broadcom Tomahawk 2) ECMP table limit. To simulate unlimited table
# size, simply set the limit to the worst case consumption:
# max port weight (200000 Mbps) * # ports/group (64) * [# src groups (64) +
# # transit groups (65*65)] = 54899200000.
TABLE_LIMIT = 16 * 1024 if not INFINITE_ECMP_TABLE else 54899200000

# Max ECMP entries a group is allowed to use. To simulate unlimited table
# size, simply set the limit to the worst case consumption:
# max port weight (200000 Mbps) * # ports/group (64) = 12800000
MAX_GROUP_SIZE = 256 if not INFINITE_ECMP_TABLE else 12800000

# True to enable a set of improved heuristics in group reduction.
# (1) pruning policy. (2) max group size. (3) table limit used. (4) group
# admission policy.
IMPROVED_HEURISTIC = False

# True to enable modified EuroSys algorithm, i.e., perform pruning.
EUROSYS_MOD = False

# Number of parallel group reductions allowed to run.
PARALLELISM = 16

# Timeout in seconds for a single Gurobi invocation.
GUROBI_TIMEOUT = 120

# The algorithm to use for group reduction.
# Must be one of eurosys[_mod]/google[_new]/carving/gurobi.
GR_ALGO = 'google_new'
