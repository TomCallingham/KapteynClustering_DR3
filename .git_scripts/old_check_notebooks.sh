#!/bin/bash
# Should loop over subdirectories!
echo "running jupytext notebook check"
git_dir=$(git rev-parse --show-toplevel)
dir=$git_dir/Notebooks
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
