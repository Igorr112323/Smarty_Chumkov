#!/usr/bin/env bash
# Положить файлы в ветку через Contents API.
#
# Почему не git: задания Actions коммитят результат сами, а в клоне на раннере
# depth=1 и sparse-checkout — `git rebase FETCH_HEAD` регулярно не сходится, и
# результат молча теряется (шаг «Сохранить» при этом выглядит успешным, потому
# что падение маскировалось). Contents API не зависит от состояния локального
# репозитория и обновляет существующий файл по его sha.
#
#   bash tools/push_file58.sh "сообщение" файл1 [файл2 ...]
set -euo pipefail
message="${1:?нужно сообщение}"; shift || true
branch="${BRANCH:-arena/4a5c459f-smarty-chumkov}"
repo="${GITHUB_REPOSITORY:?нужен GITHUB_REPOSITORY}"
pushed=0
for path in "$@"; do
  [ -f "$path" ] || { echo "нет файла $path, пропуск"; continue; }
  content=$(base64 -w0 "$path")
  sha=$(gh api "repos/${repo}/contents/${path}?ref=${branch}" --jq '.sha' 2>/dev/null || echo "")
  if [ -n "$sha" ]; then
    gh api -X PUT "repos/${repo}/contents/${path}" \
      -f message="$message" -f content="$content" -f branch="$branch" -f sha="$sha" \
      --jq '.commit.sha' >/dev/null
  else
    gh api -X PUT "repos/${repo}/contents/${path}" \
      -f message="$message" -f content="$content" -f branch="$branch" \
      --jq '.commit.sha' >/dev/null
  fi
  echo "загружено: $path ($(stat -c%s "$path") байт)"
  pushed=$((pushed+1))
done
echo "файлов загружено: $pushed"
