package(default_visibility = ["//visibility:public"])

filegroup(
    name = "testdata",
    srcs = glob([
        "data/*.textproto",
    ]),
)

py_test(
    name = 'load_toy_test',
    srcs = ['load_toy_test.py'],
    deps = [
        '//topology:topology',
        '//topology:topogen',
        '//traffic:traffic',
        '//traffic:tmgen',
        '//common:flags',
    ],
    data = glob(['data/*.textproto']),
    python_version = 'PY3',
    size = 'medium'
)

py_test(
    name = 'wcmp_alloc_test',
    srcs = ['wcmp_alloc_test.py'],
    deps = [
        '//topology:topology',
        '//topology:topogen',
        '//traffic:traffic',
        '//localTE:wcmp_alloc',
        '//localTE:group_reduction',
    ],
    data = glob(['data/*.textproto']),
    python_version = 'PY3',
    size = 'medium'
)

py_test(
    name = 'global_TE_test',
    srcs = ['global_TE_test.py'],
    deps = [
        '//topology:topology',
        '//topology:topogen',
        '//traffic:traffic',
        '//traffic:tmgen',
        '//globalTE:global_te',
    ],
    data = glob(['data/*.textproto']),
    python_version = 'PY3',
    size = 'medium'
)
