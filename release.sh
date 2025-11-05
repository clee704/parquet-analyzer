#!/bin/bash
set -euo pipefail

# Ensure clean git state; but allow untracked files
if [[ -n $(git status --porcelain --untracked-files=no) ]]; then
    echo "Git working directory is not clean. Please commit or stash changes before releasing."
    exit 1
fi

echo "Running pre-release checks..."
hatch run dev:check

current_version=$(hatch version)
echo "Current version: $current_version"
echo -n "Enter new version: "
read -r new_version

hatch version "$new_version"

git add -u .

echo "Changes to be committed:"
git diff --staged

echo -n "Commit and tag the new version? (y/n): "
read -r confirm
if [[ $confirm == "y" ]]; then
    git commit -m "chore: bump to $new_version"
    git tag "v$new_version"
    echo "Committed and tagged version $new_version"
else
    exit 1
fi

hatch clean
hatch build

ls -al dist/

echo -n "Upload to PyPI? (y/n): "
read -r upload_confirm

if [[ $upload_confirm == "y" ]]; then
    hatch publish
    echo "Uploaded to PyPI"
else
    exit 1
fi
