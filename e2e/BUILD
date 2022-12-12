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
    ],
    data = [
        "//tests:data/toy2.textproto",
        "//tests:data/toy2_traffic.textproto",
    ],
    python_version = 'PY3',
)
