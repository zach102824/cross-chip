run i#!/usr/bin/env bash
# Quick commit + push on the windows branch.
#
# Usage (from project root):
#   ./save_to_github.sh
#   ./save_to_github.sh "your commit message"
#
# In Cursor chat, say: gpush
#
# Commits on branch: windows-branch
# Pushes to origin (configured remote).
set -euo pipefail

TARGET_BRANCH="windows-branch"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Error: this folder is not a git repository."
  exit 1
fi

if git show-ref --verify --quiet "refs/heads/$TARGET_BRANCH"; then
  echo "Switching to branch '$TARGET_BRANCH'..."
  git checkout "$TARGET_BRANCH"
else
  echo "Creating branch '$TARGET_BRANCH'..."
  git checkout -b "$TARGET_BRANCH"
fi

HAS_CHANGES=0
if [[ -n "$(git status --porcelain)" ]]; then
  HAS_CHANGES=1
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
elif git rev-parse "origin/$TARGET_BRANCH" >/dev/null 2>&1; then
  AHEAD_COUNT="$(git rev-list --count "origin/$TARGET_BRANCH..HEAD")"
fi

if [[ "$HAS_CHANGES" -eq 0 && "$AHEAD_COUNT" -eq 0 ]]; then
  echo "No local commits ahead of remote. Nothing to push."
  exit 0
fi

echo "Pushing branch '$TARGET_BRANCH' to origin..."
git push -u origin "$TARGET_BRANCH"

echo "Refreshing remote tracking refs..."
git fetch origin

echo "Done. Changes are on branch '$TARGET_BRANCH'."
