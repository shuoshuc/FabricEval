# Credits: https://thethoughtfulkoala.com/posts/2020/05/08/py-protobuf-bazel.html
package(default_visibility = ["//visibility:public"])

load("@com_google_protobuf//:protobuf.bzl", "py_proto_library")

py_proto_library(
    name = "topology_proto",
    srcs = ["topology.proto"],
    deps = ["@com_google_protobuf//:protobuf_python"],
)
