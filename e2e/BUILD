package(default_visibility = ["//visibility:public"])

py_binary(
    name = 'run',
    srcs = ['run.py'],
    deps = [
        '//topology:topology',
        '//topology:topogen',
        '//traffic:traffic',
        '//traffic:tmgen',
        '//globalTE:global_te',
        '//localTE:wcmp_alloc',
        '//localTE:group_reduction',
    ],
    data = [
        "//tests:data/toy2.textproto",
        "//tests:data/toy2_traffic.textproto",
        "//tests:data/toy3_te_sol.textproto",
    ],
    python_version = 'PY3',
)
