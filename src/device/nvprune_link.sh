#!/bin/sh
#
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information
#
# Usage: nvprune_link.sh <dir> <nvprune> <cuobjdump> <readelf> -- <link command ...>
#
# Linker launcher of the libnccl.so target (cmake/NcclSharedDeviceObjects.cmake). Runs the link
# command with each device object listed in <dir>/objects.txt that has relocatable cubins to remove
# replaced, in place, by its copy without them, made by nvprune_copy.sh in <dir>/objects/ against
# the device link object of the command: the one object in it, other than the device objects, with
# a .nv_fatbin section. Nothing else in the command or its environment changes. The command is run
# unchanged if it uses a response file, if it does not have exactly one such object, or if anything
# here fails.
dir=$1
tools="$2 $3 $4"
nvprune=$2
cuobjdump=$3
readelf=$4
[ "$5" = -- ] || { echo "nvprune_link.sh: usage: $0 <dir> <nvprune> <cuobjdump> <readelf> -- <command ...>" >&2; exit 2; }
shift 5
copy_script=$(dirname "$0")/nvprune_copy.sh
skip="libnccl.so: linking the original device objects"

# Physical absolute path of an existing file (the command uses paths relative to the build directory).
phys() {
  d=$(cd -P "$(dirname "$1")" 2>/dev/null && pwd -P) || return 1
  printf '%s/%s\n' "$d" "${1##*/}"
}

for a in "$@"; do
  case "$a" in @* | *,@*) echo "$skip (the link command uses a response file)"; exec "$@" ;; esac
done
[ -f "$dir/objects.txt" ] && mkdir -p "$dir/objects" || { echo "$skip (cannot use $dir)"; exec "$@"; }
# (No "exit" inside the loops: a loop with a redirection may run in this shell.)
list="$dir/objects/list"
ok=1
while IFS= read -r o; do
  [ -z "$o" ] || phys "$o" || { ok=0; break; }
done < "$dir/objects.txt" > "$list" || ok=0
[ "$ok" = 1 ] || { echo "$skip (cannot read $dir/objects.txt)"; exec "$@"; }

# The device link object: the one object of the command, other than the device objects, with a
# .nv_fatbin section (the host objects of libnccl.so have none).
dlink=
for a in "$@"; do
  case "$a" in *.o) ;; *) continue ;; esac
  n=$(phys "$a") || continue
  grep -qxF "$n" "$list" && continue
  sections=$(LC_ALL=C "$readelf" -SW "$a" 2>/dev/null) || { echo "$skip (readelf failed on $a)"; exec "$@"; }
  case "$sections" in
    *" .nv_fatbin "*) [ -z "$dlink" ] || { echo "$skip (two objects with .nv_fatbin)"; exec "$@"; }; dlink=$n ;;
  esac
done
[ -n "$dlink" ] || { echo "$skip (no device link object)"; exec "$@"; }

# Copies are reused while newer than their object, the device link object and the scripts, and
# while the tools are the same. Only copies that differ from their object replace it.
if [ "$(cat "$dir/objects/tools" 2>/dev/null)" != "$tools" ]; then
  rm -f "$dir/objects/"*.o && printf '%s\n' "$tools" > "$dir/objects/tools" || { echo "$skip (cannot write $dir/objects)"; exec "$@"; }
fi
map="$dir/objects/map"
ok=1
while IFS= read -r o; do
  copy="$dir/objects/$(printf '%s' "$o" | cksum | cut -d ' ' -f 1)_${o##*/}"
  if [ ! "$copy" -nt "$o" ] || [ ! "$copy" -nt "$dlink" ] || [ ! "$copy" -nt "$copy_script" ] || [ ! "$copy" -nt "$0" ]; then
    NVPRUNE=$nvprune CUOBJDUMP=$cuobjdump READELF=$readelf sh "$copy_script" "$o" "$dlink" "$copy" >&2 || { ok=0; break; }
  fi
  [ -f "$copy" ] || { ok=0; break; }
  cmp -s "$copy" "$o" || printf '%s\t%s\n' "$o" "$copy"
done < "$list" > "$map" || ok=0
[ "$ok" = 1 ] || { echo "$skip (could not make the copies)"; exec "$@"; }

# The command with these objects replaced in place.
first=1
for a in "$@"; do
  [ "$first" = 0 ] || { set --; first=0; }
  case "$a" in
    *.o)
      if n=$(phys "$a"); then
        c=$(awk -F '\t' -v k="$n" '$1 == k { print $2; exit }' "$map")
        [ -z "$c" ] || a=$c
      fi ;;
  esac
  set -- "$@" "$a"
done
exec "$@"
