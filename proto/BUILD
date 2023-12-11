# Credits: https://thethoughtfulkoala.com/posts/2020/05/08/py-protobuf-bazel.html
package(default_visibility = ["//visibility:public"])

load("@protobuf//:protobuf.bzl", "py_proto_library")

py_proto_library(
    name = "topology_proto",
    srcs = ["topology.proto"],
)

py_proto_library(
    name = "traffic_proto",
    srcs = ["traffic.proto"],
)

py_proto_library(
    name = "te_solution_proto",
    srcs = ["te_solution.proto"],
)
