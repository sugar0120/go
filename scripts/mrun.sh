#!/bin/bash
# Run Python on the molab kernel. Usage: MOLAB_URL=... MOLAB_TOKEN=... mrun.sh -c "code"   (or -f file.py via -c "$(cat f)")
exec ~/.claude/skills/marimo-pair/scripts/execute-code.sh --url "$MOLAB_URL" --token "$MOLAB_TOKEN" "$@" 2>&1 | grep -v '^Warning: connecting'
