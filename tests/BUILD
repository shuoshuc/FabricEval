package(default_visibility = ["//visibility:public"])

py_test(
    name = 'load_toy',
    srcs = ['load_toy.py'],
    deps = [
        '//topology:topology',
    ],
    data = glob(['data/*.textproto']),
    python_version = 'PY3',
)
