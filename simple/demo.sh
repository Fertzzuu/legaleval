#!/usr/bin/env bash
# Builds the solution from the bare Red Hat UBI base image and runs it.
# Usage (from anywhere): simple/demo.sh
# Expects train.csv, test.csv and sample_submission.csv in ../data relative to this script.
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
data="$(cd "$here/../data" && pwd)"
image="legaleval-simple"

echo "== 1. The starting point: an empty Red Hat base image"
docker run --rm registry.access.redhat.com/ubi9/ubi-minimal:9.8 \
    sh -c 'cat /etc/redhat-release; command -v python3 || echo "no python installed"'

echo "== 2. Build the solution image from it (no cache, fresh base pull)"
docker build --no-cache --pull -t "$image" "$here"

echo "== 3. Run it: competition data mounted read-only, submission written to simple/outputs"
mkdir -p "$here/outputs"
docker run --rm \
    -v "$data:/data:ro" \
    -v "$here/outputs:/app/outputs" \
    "$image"

echo "== Done: $here/outputs/submission.csv"
