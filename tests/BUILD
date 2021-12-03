package(default_visibility = ["//visibility:public"])

py_test(
    name = 'load_toy_test',
    srcs = ['load_toy_test.py'],
    deps = [
        '//topology:topology',
    ],
    data = glob(['data/*.textproto']),
    python_version = 'PY3',
)

py_test(
    name = 'wcmp_alloc_test',
    srcs = ['wcmp_alloc_test.py'],
    deps = [
        '//topology:topology',
        '//wcmp_alloc:wcmp_alloc',
        '//wcmp_alloc:group_reduction',
    ],
    data = glob(['data/*.textproto']),
    python_version = 'PY3',
)
