#!/bin/bash
# Compatibility wrapper for tools/ops/validate_setup.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TARGET="${PROJECT_ROOT}/tools/ops/validate_setup.sh"

echo "Warning: scripts/validate_setup.sh is deprecated; use tools/ops/validate_setup.sh instead." >&2
exec bash "$TARGET" "$@"
