#!/bin/zsh

cd ./ || exit 1

folder_names=()
for dir in */; do
  folder_names+=("${dir%/}")
done

folder_names=("${(@on)folder_names:#?}")  
for file in *.csv; do
  [[ -e "$file" ]] || continue

  for folder in $folder_names; do
    if [[ "$file" == "$folder "* ]]; then
      echo "Moving '$file' to '$folder/'"
      mv -- "$file" "$folder/"
      break
    fi
  done
done

