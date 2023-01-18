package(default_visibility = ["//visibility:public"])

py_library(
    name = 'wcmp_alloc',
    srcs = ['wcmp_alloc.py'],
    deps = [
        "//proto:te_solution_proto",
        "//common:common",
    ],
)

py_library(
    name = 'group_reduction',
    srcs = ['group_reduction.py'],
    deps = [
        "//common:common",
        "//common:flags",
        "//proto:te_solution_proto",
    ],
)

py_binary(
    name = 'test_driver',
    srcs = ['test_driver.py'],
    deps = [
        "//common:flags",
        '//localTE:group_reduction',
    ],
    python_version = 'PY3',
)
