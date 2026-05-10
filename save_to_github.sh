#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Error: this folder is not a git repository."
  exit 1
fi

if [[ -z "$(git status --porcelain)" ]]; then
  echo "No changes to commit. Working tree is clean."
  exit 0
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

echo "Staging all changes..."
git add -A

echo "Creating commit..."
git commit -m "$COMMIT_MESSAGE"

echo "Pushing branch '$BRANCH'..."
if git rev-parse --abbrev-ref --symbolic-full-name "@{u}" >/dev/null 2>&1; then
  git push
else
  git push -u origin "$BRANCH"
fi

echo "Done. Changes are pushed to GitHub."
