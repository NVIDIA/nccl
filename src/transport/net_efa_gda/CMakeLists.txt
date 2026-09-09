# net_efa_gda transport
add_subdirectory(efa-dp-direct)

# Add EFA-DP-Direct device headers to parent scope
set(DEVICE_EFA_DP_HEADERS ${DEVICE_EFA_DP_HEADERS} PARENT_SCOPE)
