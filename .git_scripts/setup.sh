#!/bin/bash
git_dir=$(git rev-parse --show-toplevel)
file_post_merge="$git_dir"/.git/hooks/post-merge
script_file_post_merge="$git_dir"/.git_scripts/post-merge
cp "$script_file_post_merge" "$file_post_merge"
chmod +x "$file_post_merge"
bash "$file_post_merge"
