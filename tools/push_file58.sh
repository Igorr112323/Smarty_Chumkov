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
  # 80 заданий пишут в одну ветку одновременно: PUT со старым sha получает 409
  # «does not match», поэтому перечитываем sha и повторяем.
  ok=0
  for attempt in 1 2 3 4 5 6 7 8; do
    # ВАЖНО: при 404 gh печатает JSON ошибки в stdout, поэтому `|| echo ""` не
    # спасает: в sha попадал текст {"message":"Not Found"...}, и PUT с этим sha
    # всегда получал 409. Проверяем, что ответ — настоящий 40-символьный хеш.
    sha=$(gh api "repos/${repo}/contents/${path}?ref=${branch}" --jq '.sha' 2>/dev/null || true)
    case "$sha" in
      [0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]*) ;;
      *) sha="" ;;
    esac
    if [ -n "$sha" ] && [ "${#sha}" -ne 40 ]; then sha=""; fi
    if [ -n "$sha" ]; then
      if gh api -X PUT "repos/${repo}/contents/${path}" -f message="$message" \
           -f content="$content" -f branch="$branch" -f sha="$sha" \
           --jq '.commit.sha' >/dev/null 2>&1; then ok=1; break; fi
    else
      if gh api -X PUT "repos/${repo}/contents/${path}" -f message="$message" \
           -f content="$content" -f branch="$branch" \
           --jq '.commit.sha' >/dev/null 2>&1; then ok=1; break; fi
    fi
    echo "попытка $attempt не прошла ($path), повтор через $((attempt * 2)) с"
    sleep $((attempt * 2))
  done
  [ "$ok" = "1" ] || { echo "НЕ УДАЛОСЬ загрузить $path"; exit 1; }
  echo "загружено: $path ($(stat -c%s "$path") байт)"
  pushed=$((pushed+1))
done
echo "файлов загружено: $pushed"
