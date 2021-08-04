package(default_visibility = ["//visibility:public"])

py_test(
    name = 'load_testnet',
    srcs = ['load_testnet.py'],
    deps = [
        '//topology:topology',
    ],
    data = [
        ':data/testnet_topo.textproto'
    ],
    python_version = 'PY3',
)
