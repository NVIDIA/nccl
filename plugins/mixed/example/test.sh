#!/bin/sh
#
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information
#

NCCL_SRCDIR=../../../src

# Check plugin versions
netplugin_ver=$(cat ${NCCL_SRCDIR}/include/plugin/nccl_net.h | grep NCCL_NET_PLUGIN_SYMBOL | cut -d" " -f3)
tunplugin_ver=$(cat ${NCCL_SRCDIR}/include/plugin/nccl_tuner.h | grep NCCL_TUNER_PLUGIN_SYMBOL | cut -d" " -f3)

net_symbols=$(nm -D libnccl-mixed.so | cut -d" " -f3 | grep NetPlugin | sort -k 15 -r)
tun_symbols=$(nm -D libnccl-mixed.so | cut -d" " -f3 | grep TunerPlugin | sort -k 17 -r)

net_ok=0
tun_ok=0

# Search for available plugin versions starting from the latest
for sym in $net_symbols ; do
  if [ "$netplugin_ver" == "$sym" ] ; then
    net_ok=1
    break
  fi
done

# Search for available plugin versions starting from the latest
for sym in $tun_symbols ; do
  if [ "$tunplugin_ver" == \"$sym\" ] ; then
    tun_ok=1
    break
  fi
done

printf "PLUGIN-NET:        Expecting symbol \"$netplugin_ver\"   "
if [ "$net_ok" -ne "1" ] ; then
  printf "[FAIL]\n"
else
  printf "[SUCCESS]\n"
fi

printf "PLUGIN-TUNER:      Expecting symbol $tunplugin_ver "
if [ "$tun_ok" -ne "1" ] ; then
  printf "[FAIL]\n"
else
  printf "[SUCCESS]\n"
fi

if [ "$net_ok" -ne "1" ] || [ "$tun_ok" -ne "1" ] ; then
  exit 1
fi
