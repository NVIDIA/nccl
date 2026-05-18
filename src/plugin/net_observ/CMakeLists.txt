# Net observ plugin sources

find_package(Protobuf REQUIRED)
find_program(GRPC_CPP_PLUGIN grpc_cpp_plugin REQUIRED)

set(RUIJIE_JSON_PROTO ${CMAKE_CURRENT_SOURCE_DIR}/ruijie-json.proto)
set(RUIJIE_JSON_GEN_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(RUIJIE_JSON_GEN_HDR_DIR ${CMAKE_SOURCE_DIR}/src/include/plugin/net_observ)

set(RUIJIE_JSON_PB_HDR ${RUIJIE_JSON_GEN_HDR_DIR}/ruijie-json.pb.h)
set(RUIJIE_JSON_PB_SRC ${RUIJIE_JSON_GEN_SRC_DIR}/ruijie-json.pb.cc)
set(RUIJIE_JSON_GRPC_HDR ${RUIJIE_JSON_GEN_HDR_DIR}/ruijie-json.grpc.pb.h)
set(RUIJIE_JSON_GRPC_SRC ${RUIJIE_JSON_GEN_SRC_DIR}/ruijie-json.grpc.pb.cc)

add_custom_command(
  OUTPUT ${RUIJIE_JSON_PB_SRC} ${RUIJIE_JSON_PB_HDR} ${RUIJIE_JSON_GRPC_SRC} ${RUIJIE_JSON_GRPC_HDR}
  COMMAND ${CMAKE_COMMAND} -E make_directory ${RUIJIE_JSON_GEN_HDR_DIR}
  COMMAND ${Protobuf_PROTOC_EXECUTABLE}
    ARGS --proto_path=${CMAKE_CURRENT_SOURCE_DIR}
         --cpp_out=${RUIJIE_JSON_GEN_SRC_DIR}
         --grpc_out=${RUIJIE_JSON_GEN_SRC_DIR}
         --plugin=protoc-gen-grpc=${GRPC_CPP_PLUGIN}
         ${RUIJIE_JSON_PROTO}
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${RUIJIE_JSON_GEN_SRC_DIR}/ruijie-json.pb.h
    ${RUIJIE_JSON_PB_HDR}
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${RUIJIE_JSON_GEN_SRC_DIR}/ruijie-json.grpc.pb.h
    ${RUIJIE_JSON_GRPC_HDR}
  COMMAND ${CMAKE_COMMAND} -E remove
    ${RUIJIE_JSON_GEN_SRC_DIR}/ruijie-json.pb.h
    ${RUIJIE_JSON_GEN_SRC_DIR}/ruijie-json.grpc.pb.h
  DEPENDS ${RUIJIE_JSON_PROTO}
  COMMENT "Compiling ruijie-json.proto"
)

set(PLUGIN_NET_OBSERV_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/net_observ.cc
    ${RUIJIE_JSON_PB_SRC}
    ${RUIJIE_JSON_GRPC_SRC}
)

set(PLUGIN_NET_OBSERV_INCLUDES ${RUIJIE_JSON_GEN_HDR_DIR} PARENT_SCOPE)

# Add net observ plugin sources to parent scope
set(PLUGIN_NET_OBSERV_SOURCES ${PLUGIN_NET_OBSERV_SOURCES} PARENT_SCOPE)
