#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
checkin_cmd="$script_dir/checkin"
converted_dir="./converted"
mkdir -p "$converted_dir"

lock_file="./.split_multipage_pdfs.lock"
exec 9>"$lock_file"
if ! flock -n 9; then
  echo "split_multipage_pdfs.sh is already running; exiting."
  exit 0
fi

shopt -s nullglob
shopt -s nocaseglob
processed_total=false

while true; do
  split_files=()

  for pdf in ./*.pdf; do
    base_name="${pdf##*/}"

    # Skip already generated split outputs if rerun.
    if [[ "${base_name,,}" =~ _[0-9]{3}\.pdf$ ]]; then
      continue
    fi

    pages="$(pdfinfo "$pdf" 2>/dev/null | awk -F': *' '/^Pages:/ {print $2}')"

    if [[ -z "${pages:-}" ]]; then
      echo "Skipping unreadable PDF: $pdf" >&2
      continue
    fi

    if (( pages <= 1 )); then
      echo "Skipping single-page PDF: $pdf"
      continue
    fi

    stem="${pdf%.*}"
    output_pattern="${stem}_%03d.pdf"

    echo "Splitting: $pdf ($pages pages)"
    pdfseparate "$pdf" "$output_pattern"

    for (( page_num = 1; page_num <= pages; page_num++ )); do
      printf -v split_file "%s_%03d.pdf" "$stem" "$page_num"
      split_files+=("${split_file##*/}")
    done

    mv "$pdf" "$converted_dir/"
    echo "Moved original to: $converted_dir/$base_name"
  done

  if (( ${#split_files[@]} == 0 )); then
    break
  fi

  processed_total=true
  echo "Split ${#split_files[@]} PDF page(s); skipping checkin (disabled during archive backfill)."
  # "$checkin_cmd" process "${split_files[@]}"
done

if [[ "$processed_total" == false ]]; then
  echo "No multi-page PDFs found to split."
fi
