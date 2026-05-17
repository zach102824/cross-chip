#!/usr/bin/env bash
# Quick commit + push for cross_chips_sim.
#
# Usage (from project root):
#   ./save_to_github.sh
#   ./save_to_github.sh "your commit message"
#
# In Cursor chat, say: gpush
#
# Pushes to: https://github.com/zach102824/cross-chip
# Uses explicit repo URL (reliable when `git push` via origin times out).
set -euo pipefail

GITHUB_REPO="https://github.com/zach102824/cross-chip.git"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Error: this folder is not a git repository."
  exit 1
fi

HAS_CHANGES=0
if [[ -n "$(git status --porcelain)" ]]; then
  HAS_CHANGES=1
fi

BRANCH="$(git branch --show-current)"
if [[ -z "$BRANCH" ]]; then
  echo "Error: could not detect current branch."
  exit 1
fi

if [[ $# -gt 0 ]]; then
  COMMIT_MESSAGE="$*"
else
  COMMIT_MESSAGE="Update project files ($(date '+%Y-%m-%d %H:%M:%S'))"
fi

if [[ "$HAS_CHANGES" -eq 1 ]]; then
  echo "Staging all changes..."
  git add -A

  echo "Creating commit..."
  git commit -m "$COMMIT_MESSAGE"
else
  echo "No new file changes to commit."
fi

AHEAD_COUNT=0
if git rev-parse --abbrev-ref --symbolic-full-name "@{u}" >/dev/null 2>&1; then
  AHEAD_COUNT="$(git rev-list --count "@{u}..HEAD")"
elif git rev-parse "origin/$BRANCH" >/dev/null 2>&1; then
  AHEAD_COUNT="$(git rev-list --count "origin/$BRANCH..HEAD")"
fi

if [[ "$HAS_CHANGES" -eq 0 && "$AHEAD_COUNT" -eq 0 ]]; then
  echo "No local commits ahead of remote. Nothing to push."
  exit 0
fi

echo "Pushing branch '$BRANCH' to $GITHUB_REPO..."
git push "$GITHUB_REPO" "$BRANCH:$BRANCH"

echo "Refreshing remote tracking refs..."
git fetch origin

echo "Done. Changes are on GitHub: https://github.com/zach102824/cross-chip"
