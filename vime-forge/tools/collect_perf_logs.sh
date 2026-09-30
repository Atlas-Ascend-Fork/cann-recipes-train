#!/bin/bash
# Collect Ray worker logs from multiple nodes for offline perf timeline rendering.
#
# Usage:
#   bash tools/collect_perf_logs.sh <driver_log> <output_dir> <node1> [node2] [node3] ...
#
# Example:
#   bash tools/collect_perf_logs.sh /mnt/workspace/venv_logs/launch.log ./collected_logs 192.169.87.91 192.169.87.92
#
# Then render:
#   python tools/build_perf_timeline.py /mnt/workspace/venv_logs/launch.log ./collected_logs/*/*.out --html out.html --trace out.json

set -euo pipefail

if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <driver_log> <output_dir> <node1> [node2] [node3] ..."
    echo ""
    echo "Arguments:"
    echo "  driver_log   Path to the driver launch log on the local machine"
    echo "  output_dir   Directory to store collected logs"
    echo "  node1...     SSH hostnames or IPs of Ray worker nodes"
    echo ""
    echo "Example:"
    echo "  $0 /mnt/workspace/venv_logs/launch.log ./collected_logs 192.169.87.91 192.169.87.92"
    exit 1
fi

DRIVER_LOG="$1"
OUTPUT_DIR="$2"
shift 2
NODES=("$@")

mkdir -p "$OUTPUT_DIR"

# Copy driver log
cp "$DRIVER_LOG" "$OUTPUT_DIR/driver.log"
echo "[1/2] Driver log copied to $OUTPUT_DIR/driver.log"

# Find Ray session directory from local /tmp/ray
SESSION=$(ls -t /tmp/ray/ 2>/dev/null | grep '^session_' | head -1 || true)

if [ -z "$SESSION" ]; then
    echo "Error: No Ray session found in local /tmp/ray/."
    echo "If driver is on another node, specify its logs manually."
    exit 1
fi

echo "Ray session: $SESSION"

# Collect .out and .err files from each node
COLLECTED_COUNT=0
LOCAL_HOSTNAME=$(hostname)

collect_remote_node() {
    local node="$1"
    local node_dir="$2"
    mkdir -p "$node_dir"
    echo "  Connecting to $node..."
    echo "  (will prompt for password if SSH key not configured)"
    # Use scp directly - it will prompt for password interactively
    scp -o StrictHostKeyChecking=no \
        "$node:/tmp/ray/$SESSION/logs/worker-*.out" \
        "$node_dir/" 2>/dev/null || true
    scp -o StrictHostKeyChecking=no \
        "$node:/tmp/ray/$SESSION/logs/worker-*.err" \
        "$node_dir/" 2>/dev/null || true
}

for node in "${NODES[@]}"; do
    NODE_DIR="$OUTPUT_DIR/$node"
    mkdir -p "$NODE_DIR"
    echo "[2/2] Collecting logs from $node..."

    # Skip SSH if this is the local node
    if [ "$node" = "$LOCAL_HOSTNAME" ] || [ "$node" = "localhost" ] || [ "$node" = "127.0.0.1" ]; then
        cp /tmp/ray/$SESSION/logs/worker-*.out "$NODE_DIR/" 2>/dev/null || true
        cp /tmp/ray/$SESSION/logs/worker-*.err "$NODE_DIR/" 2>/dev/null || true
    else
        collect_remote_node "$node" "$NODE_DIR"
    fi

    # Filter to only files containing VIME_PERF_TRACE
    for f in "$NODE_DIR"/*.out "$NODE_DIR"/*.err; do
        [ -f "$f" ] || continue
        if ! grep -q "VIME_PERF_TRACE" "$f" 2>/dev/null; then
            rm "$f"
        fi
    done

    NODE_COUNT=$(find "$NODE_DIR" -name "*.out" -o -name "*.err" 2>/dev/null | wc -l)
    COLLECTED_COUNT=$((COLLECTED_COUNT + NODE_COUNT))
    echo "  Collected $NODE_COUNT log files from $node"
done

# Also collect local worker logs if not already covered
LOCAL_DIR="$OUTPUT_DIR/$LOCAL_HOSTNAME"
if [ ! -d "$LOCAL_DIR" ]; then
    mkdir -p "$LOCAL_DIR"
    cp /tmp/ray/$SESSION/logs/worker-*.out "$LOCAL_DIR/" 2>/dev/null || true
    cp /tmp/ray/$SESSION/logs/worker-*.err "$LOCAL_DIR/" 2>/dev/null || true
    for f in "$LOCAL_DIR"/*.out "$LOCAL_DIR"/*.err; do
        [ -f "$f" ] || continue
        if ! grep -q "VIME_PERF_TRACE" "$f" 2>/dev/null; then
            rm "$f"
        fi
    done
    LOCAL_COUNT=$(find "$LOCAL_DIR" -name "*.out" -o -name "*.err" 2>/dev/null | wc -l)
    COLLECTED_COUNT=$((COLLECTED_COUNT + LOCAL_COUNT))
    echo "  Collected $LOCAL_COUNT log files from local node"
fi

echo ""
echo "Done! Collected $COLLECTED_COUNT log files to $OUTPUT_DIR/"
echo ""
echo "Render with:"
echo "  python tools/build_perf_timeline.py \\"
echo "    $OUTPUT_DIR/driver.log \\"
echo "    $OUTPUT_DIR/*/*.out \\"
echo "    $OUTPUT_DIR/*/*.err \\"
echo "    --html out.html --trace out.json"
