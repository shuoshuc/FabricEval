package(default_visibility = ["//visibility:public"])

py_library(
    name = 'topology',
    srcs = ['topology.py'],
    deps = [
        "//proto:topology_proto",
        "//proto:te_solution_proto",
        "//topology:graph_db",
    ],
)

py_library(
    name = 'topogen',
    srcs = ['topogen.py'],
    deps = [
        "//proto:topology_proto",
        "//common:flags",
        "//topology:striping_plan",
    ],
)

py_library(
    name = 'striping_plan',
    srcs = ['striping_plan.py'],
    deps = [
        "//common:common",
        "//common:flags",
    ],
)

py_library(
    name = 'graph_db',
    srcs = ['graph_db.py'],
    deps = [
        "//common:common",
        "//common:flags",
    ],
)
