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
    echo "  git add -A && git commit -m '<initial commit>'"
    if gh repo view "$repo_id" >/dev/null 2>&1; then
        echo "  # repository already exists, nothing to create"
    else
        echo "  gh repo create $repo_id $visibility --description \"$DESCRIPTION\""
    fi
    echo "  git remote add origin https://github.com/$repo_id.git   # if origin is missing"
    echo "  git push -u origin main"
    echo "  gh repo edit $repo_id --description \"...\" --add-topic $TOPICS"
    exit 0
fi

cd "$root"

if [[ ! -d .git ]]; then
    git init -b main
fi

git add -A
if ! git diff --cached --quiet; then
    git commit -m "${NAME} ROCm port: gfx1200/gfx1201 (RDNA4) and gfx115x APUs"
fi

# Create the repository first (empty, no remote), then wire up origin and push.
# Letting `gh repo create --source . --remote origin --push` do it fails when
# origin already exists, which is exactly what happens on a second run.
if gh repo view "$repo_id" >/dev/null 2>&1; then
    echo "repository exists: https://github.com/$repo_id"
else
    echo "creating repository"
    gh repo create "$repo_id" "$visibility" --description "$DESCRIPTION"
fi

if git remote get-url origin >/dev/null 2>&1; then
    echo "using existing origin: $(git remote get-url origin)"
    if [[ "$(git remote get-url origin)" != *"github.com/${repo_id}"* ]]; then
        echo "warning: origin does not point at $repo_id; pushing there anyway" >&2
    fi
else
    git remote add origin "https://github.com/${repo_id}.git"
fi

# Make sure git can authenticate over https with the gh credentials.
gh auth setup-git >/dev/null 2>&1 || true

git push -u origin main

gh repo edit "$repo_id" --description "$DESCRIPTION" --add-topic "$TOPICS" >/dev/null

echo
echo "done: https://github.com/$repo_id"
