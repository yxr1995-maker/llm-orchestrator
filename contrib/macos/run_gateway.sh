#!/bin/bash
# llm-orchestrator launch script (for launchd)
# assume the repo is cloned at ~/llm-orchestrator with a .venv created
set -e
cd "$(dirname "$0")/../.."
exec .venv/bin/python -m app.main
