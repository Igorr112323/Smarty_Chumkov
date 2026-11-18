#!/usr/bin/env bash
# Закоммитить файлы отчёта в ветку: артефакты Actions из песочницы недоступны,
# поэтому единственный надёжный канал — коммит в ветку.
set -euo pipefail

message="${1:?нужно сообщение коммита}"
shift || true
branch="${BRANCH:-arena/4a5c459f-smarty-chumkov}"

git config user.name github-actions
git config user.email 41898282+github-actions[bot]@users.noreply.github.com
shopt -s nullglob
files=("$@")
existing=()
for pattern in "${files[@]}"; do
  for file in $pattern; do
    [ -f "$file" ] && existing+=("$file")
  done
done
if [ ${#existing[@]} -eq 0 ]; then
  echo "нечего коммитить: файлы не появились"
  exit 0
fi
git add --sparse -f "${existing[@]}"
git commit -m "$message" || { echo "коммит пуст"; exit 0; }
for attempt in 1 2 3; do
  git fetch origin "$branch"
  if git rebase FETCH_HEAD && git push origin HEAD:"$branch"; then
    echo "запушено"
    exit 0
  fi
  sleep 10
done
echo "ОШИБКА: не удалось запушить"
exit 1
