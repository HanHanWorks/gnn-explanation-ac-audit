#!/usr/bin/env bash
# Regenerate Figure 1 (PDF) from the vector source.
# Requires rsvg-convert (librsvg2) and the Times New Roman font.
set -euo pipefail
cd "$(dirname "$0")/.."
rsvg-convert -f pdf -o figures/figure_1_workflow.pdf figures/Figure1_source.svg
