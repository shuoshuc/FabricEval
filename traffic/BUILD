package(default_visibility = ["//visibility:public"])

filegroup(
    name = "tracedata",
    srcs = glob([
        "data/*.txt",
    ]),
)

py_library(
    name = 'traffic',
    srcs = ['traffic.py'],
    deps = [
        "//proto:traffic_proto",
    ],
)

py_library(
    name = 'tmgen',
    srcs = ['tmgen.py'],
    deps = [
        "//proto:traffic_proto",
        "//common:flags",
    ],
)
