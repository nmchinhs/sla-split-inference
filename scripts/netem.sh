#!/usr/bin/env bash
# Shape a device container's egress with tc netem. Run from the host.
#
#   scripts/netem.sh <container> <rate_mbit> <delay_ms> [loss_pct]
#   scripts/netem.sh <container> clear
#
# The container needs cap_add: [NET_ADMIN] and iproute2 (both set up for device-*).
set -euo pipefail

IFACE="${NETEM_IFACE:-eth0}"

usage() {
    sed -n '2,6p' "$0" | sed 's/^# \{0,1\}//' >&2
    exit 2
}

[[ $# -ge 2 ]] || usage
container="$1"

if [[ "$2" == "clear" ]]; then
    # No qdisc to delete is not an error: the link is already unshaped.
    docker exec "$container" tc qdisc del dev "$IFACE" root 2>/dev/null || true
    echo "$container: netem cleared on $IFACE"
    exit 0
fi

[[ $# -ge 3 && $# -le 4 ]] || usage
rate_mbit="$2"
delay_ms="$3"
loss_pct="${4:-0}"

docker exec "$container" tc qdisc replace dev "$IFACE" root netem \
    rate "${rate_mbit}mbit" delay "${delay_ms}ms" loss "${loss_pct}%"
echo "$container: ${rate_mbit} Mbit/s, ${delay_ms} ms delay, ${loss_pct}% loss on $IFACE"
