package(default_visibility = ["//visibility:public"])

py_binary(
    name = 'global_te',
    srcs = ['global_te.py'],
    deps = [
        "//proto:topology_proto",
        "//proto:traffic_proto",
        "//proto:te_solution_proto",
        '//topology:topology',
        '//traffic:traffic',
    ],
    data = [
        "//tests:data/toy2.textproto",
        "//tests:data/toy2_traffic.textproto",
    ],
    python_version = 'PY3',
)
