#!/bin/bash
echo "running jupytext notebook check"
git_dir=$(git rev-parse --show-toplevel )
dirs=$(find -L "$git_dir"  -maxdepth 3 -type d -not -path '*/.*' -not -path '*/__pycache__*')
# dir=$git_dir/Notebooks
for dir in $dirs; do
if [ -d "$dir"/.jupy/ ]; then
				py_files=$(ls "$dir"/.jupy/*.py)
				for py_path in $py_files; do
								py_file=$(basename -- "$py_path")
								file_name="${py_file%.*}"
								ipynb_path="$dir"/"$file_name".ipynb
								if [[ ! -f "$ipynb_path" ]]; then
												echo "$ipynb_path does not exists, making"
												jupytext --to notebook "$py_path"
												new_ipynb_path="$dir"/.jupy/"$file_name".ipynb
												mv "$new_ipynb_path" "$ipynb_path"
								fi

				done
				for py_path in $py_files; do
								touch "$py_path"
				done
fi
				done
