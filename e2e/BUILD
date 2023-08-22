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
        "//tests:testdata",
    ],
    python_version = 'PY3',
)

py_binary(
    name = 'tracegen',
    srcs = ['tracegen.py'],
    deps = [
        '//traffic:tmgen',
        "//common:flags",
    ],
    data = [
        "//traffic:tracedata",
    ],
    python_version = 'PY3',
)
