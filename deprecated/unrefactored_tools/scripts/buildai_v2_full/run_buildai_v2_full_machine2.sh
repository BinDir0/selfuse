#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MACHINE_ID=1 exec "${SCRIPT_DIR}/run_buildai_v2_full_machine.sh"
