package(default_visibility = ["//visibility:public"])

py_library(
    name = 'topology',
    srcs = ['topology.py'],
    deps = [
        "//proto:topology_proto",
        "//proto:te_solution_proto",
    ],
)

py_library(
    name = 'topogen',
    srcs = ['topogen.py'],
    deps = [
        "//proto:topology_proto",
        "//common:flags",
    ],
)
