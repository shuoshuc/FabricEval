package(default_visibility = ["//visibility:public"])

py_library(
    name = 'topology',
    srcs = ['topology.py'],
    deps = [
        "//proto:topology_proto",
    ],
)

py_binary(
    name = 'topogen',
    srcs = ['topogen.py'],
    deps = [
        "//proto:topology_proto",
    ],
)
