#!/bin/sh
#
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information
#
# Usage: nvprune_copy.sh <object> <device link object> <output>
#
# Writes the copy of a device object that libnccl.so is linked from: the object with only its
# pre-link relocatable cubins removed. An object compiled with -rdc carries them in __nv_relfatbin
# as the input of the device link; libnccl.so would need them only for JIT linking at load time,
# when the device link has no linked image for the GPU. The copy is made with
# nvprune --no-relocatable-elf, given the targets that cuobjdump lists in the object, and is used
# only if it then has exactly the PTX entries of the object, in the same order and with the same
# contents, and nothing else. Otherwise the
# object is copied as is: when it has no relocatable cubins, has entries other than relocatable
# cubins and PTX (LTO IR, for example) or also a .nv_fatbin section, when the device link has no
# linked image for one of the architectures of its relocatable cubins, or when a tool fails.
# Tools: $NVPRUNE, $CUOBJDUMP, $READELF.
LC_ALL=C
export LC_ALL
obj=$1
dlink=$2
out=$3
NVPRUNE=${NVPRUNE:-nvprune}
CUOBJDUMP=${CUOBJDUMP:-cuobjdump}
READELF=${READELF:-readelf}
tmp="$out.tmp.$$"
ptxdir="$out.ptx.$$"
trap 'rm -rf "$tmp" "$ptxdir"' EXIT
trap 'exit 1' INT TERM HUP

# Copy the object as is (with the reason, if any, on stdout).
original() {
  [ -z "$1" ] || echo "libnccl.so: using the original $obj ($1)"
  cp "$obj" "$tmp" && mv -f "$tmp" "$out" || exit 1
  exit 0
}

# Fatbin entries of a file in order, one "<kind> <arch>" line each (kind: elf, ptx, nvvm, ...);
# fails on a record it cannot read.
entries() {
  listing=$("$CUOBJDUMP" "$1" 2>/dev/null) || return 1
  printf '%s\n' "$listing" | awk '
    /^Fatbin/ { if ($0 !~ /^Fatbin [a-z]+ code:$/ || kind != "") bad = 1; kind = $2; next }
    /^arch = / { if (kind == "") bad = 1; else print kind, $3; kind = "" }
    END { if (kind != "" || bad) exit 1 }'
}

# Checksums of the PTX entries of a file (cuobjdump -xptx), sorted: ptx_sums <file> <dir>
ptx_sums() {
  case "$1" in /*) f=$1 ;; *) f=$PWD/$1 ;; esac
  mkdir -p "$2" && (cd "$2" && "$CUOBJDUMP" -xptx all "$f" > /dev/null 2>&1) || return 1
  for x in "$2"/*.ptx; do [ ! -e "$x" ] || cksum < "$x"; done | sort
}

sections=$("$READELF" -SW "$obj" 2>/dev/null) || original "readelf failed"
case "$sections" in *" __nv_relfatbin "*) ;; *) original "" ;; esac
case "$sections" in *" .nv_fatbin "*) original "it also has .nv_fatbin" ;; esac
before=$(entries "$obj") || original "cuobjdump cannot list its entries"
linked=$(entries "$dlink") || original "cuobjdump cannot list the entries of $dlink"
relocatable=$(printf '%s\n' "$before" | awk '$1 == "elf" { print $2 }' | sort -u)
ptx=$(printf '%s\n' "$before" | awk '$1 == "ptx" { print $2 }' | sort -u)
[ -n "$relocatable" ] || original ""
[ -z "$(printf '%s\n' "$before" | awk 'NF && $1 != "elf" && $1 != "ptx"')" ] || original "entries other than relocatable cubins and PTX"
linked=" $(printf '%s\n' "$linked" | awk '$1 == "elf" { printf "%s ", $2 }')"
missing=
targets=
for a in $relocatable; do
  case "$linked" in *" $a "*) ;; *) missing="$missing $a" ;; esac
  targets="$targets --generate-code=arch=compute_${a#sm_},code=$a"
done
[ -z "$missing" ] || original "no linked image for$missing"
for a in $ptx; do
  targets="$targets --generate-code=arch=compute_${a#sm_},code=compute_${a#sm_}"
done
"$NVPRUNE" --no-relocatable-elf $targets "$obj" -o "$tmp" 2>/dev/null || original "nvprune failed"
after=$(entries "$tmp") || original "cuobjdump cannot list the entries of the copy"
[ "$after" = "$(printf '%s\n' "$before" | awk '$1 != "elf"')" ] || original "the copy does not have exactly its PTX entries"
if [ -n "$ptx" ]; then
  sums=$(ptx_sums "$obj" "$ptxdir/a") && [ "$sums" = "$(ptx_sums "$tmp" "$ptxdir/b")" ] ||
    original "the PTX of the copy differs"
fi
mv -f "$tmp" "$out" || exit 1
