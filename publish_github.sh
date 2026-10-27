#!/usr/bin/env bash
# Publish this tree to GitHub with the gh CLI.
#
#   ./publish_github.sh                          # <your-user>/gsq-rocm, public
#   ./publish_github.sh efeisot/gsq-rocm         # explicit repo id
#   ./publish_github.sh efeisot/gsq-rocm --private
#   ./publish_github.sh --dry-run                # print the plan only
#
# The script:
#   1. initialises a git repo (branch main) and commits the tree,
#   2. creates the GitHub repo if it does not exist,
#   3. pushes, then sets the description and topics.
#
# It never rewrites history: an existing "origin" is reused and a normal
# `git push -u origin main` is performed.
set -euo pipefail

NAME="gsq-rocm"
DESCRIPTION="GSQ (Gumbel-Softmax Quantization) ported to AMD ROCm: gfx1200/gfx1201 (RDNA4) and gfx115x (RDNA3.5 APUs)"
TOPICS="gsq,rocm,amd,rdna4,gfx1200,quantization,llm,pytorch,hip,mixed-precision"

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_id=""
visibility="--public"
dry_run=0

usage() {
    echo "usage: $0 [owner/repo] [--private] [--dry-run]" >&2
    exit 2
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --private) visibility="--private" ;;
        --public)  visibility="--public" ;;
        --dry-run) dry_run=1 ;;
        -h|--help) usage ;;
        -*) echo "unknown option: $1" >&2; usage ;;
        *)  repo_id="$1" ;;
    esac
    shift
done

command -v gh >/dev/null 2>&1 || { echo "error: gh CLI not found (https://cli.github.com)" >&2; exit 1; }
command -v git >/dev/null 2>&1 || { echo "error: git not found" >&2; exit 1; }

if ! gh auth status >/dev/null 2>&1; then
    echo "error: not logged in to GitHub" >&2
    echo "run: gh auth login" >&2
    exit 1
fi

if [[ -z "$repo_id" ]]; then
    user="$(gh api user -q .login)"
    repo_id="${user}/${NAME}"
fi

echo "repo:     $repo_id"
echo "folder:   $root"
echo "visibility: ${visibility#--}"
echo

if [[ "$dry_run" -eq 1 ]]; then
    echo "would run:"
    echo "  git init -b main"
    echo "  git add -A && git commit -m 'GSQ ROCm port'"
    if gh repo view "$repo_id" >/dev/null 2>&1; then
        echo "  git remote add origin https://github.com/$repo_id.git   # repo exists"
        echo "  git push -u origin main"
    else
        echo "  gh repo create $repo_id --source . $visibility --push --description \"$DESCRIPTION\""
    fi
    echo "  gh repo edit $repo_id --description \"$DESCRIPTION\" --add-topic $TOPICS"
    exit 0
fi

cd "$root"

if [[ ! -d .git ]]; then
    git init -b main
fi

git add -A
if ! git diff --cached --quiet; then
    git commit -m "GSQ ROCm port: gfx1200/gfx1201 (RDNA4) and gfx115x APUs"
fi

if git remote get-url origin >/dev/null 2>&1; then
    echo "using existing origin: $(git remote get-url origin)"
else
    git remote add origin "https://github.com/${repo_id}.git"
fi

if gh repo view "$repo_id" >/dev/null 2>&1; then
    echo "repository exists; pushing"
    git push -u origin main
else
    echo "creating repository"
    gh repo create "$repo_id" --source . "$visibility" --remote origin \
        --push --description "$DESCRIPTION"
fi

gh repo edit "$repo_id" --description "$DESCRIPTION" --add-topic "$TOPICS" >/dev/null

echo
echo "done: https://github.com/$repo_id"
