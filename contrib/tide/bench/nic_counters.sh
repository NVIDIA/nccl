#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 Kuaishou Technology. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Snapshot / diff mlx5 rail-NIC counters that expose RoCE congestion (ECN marks, CNPs, PFC pause,
# retransmits) plus port byte counters.  usage: nic_counters.sh snap <file> | diff <before> <after>
# RAILS: space-separated device globs (default: bonded mlx5_bond_1x and single-port mlx5_1x).
set -u
snap() {
  for g in ${RAILS:-mlx5_bond_1[0-7] mlx5_1[0-7]}; do for d in /sys/class/infiniband/$g; do
    [ -d "$d" ] || continue
    n=$(basename "$d")
    for c in port_rcv_data port_xmit_data; do echo "$n.$c $(cat "$d/ports/1/counters/$c" 2>/dev/null || echo 0)"; done
    for c in np_ecn_marked_roce_packets np_cnp_sent rp_cnp_handled rp_cnp_ignored out_of_sequence packet_seq_err \
             local_ack_timeout_err rx_icrc_encapsulated roce_slow_restart; do
      echo "$n.$c $(cat "$d/ports/1/hw_counters/$c" 2>/dev/null || echo 0)"
    done
  done; done
  for b in /proc/net/bonding/bond[1-8]; do
    [ -f "$b" ] || continue
    for s in $(awk '/Slave Interface/ {print $3}' "$b"); do
      ethtool -S "$s" 2>/dev/null | awk -v s="$s" '/rx_prio[0-9]_pause:|tx_prio[0-9]_pause:|rx_pause:|tx_pause:|rx_discards_phy:/ {gsub(":","",$1); print s"."$1, $2}'
    done
  done
}
case "${1:-}" in
  snap) snap > "${2:?file}" ;;
  diff) join "${2:?before}" "${3:?after}" | awk '{d=$3-$2; if (d!=0) printf "%-52s %+d\n", $1, d}' ;;
  *) echo "usage: $0 snap <file> | diff <before> <after>"; exit 2 ;;
esac
