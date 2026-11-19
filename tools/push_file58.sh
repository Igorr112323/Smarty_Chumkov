#!/usr/bin/env bash
# Положить файлы в ветку через Contents API.
#
# Зачем не git: задания Actions коммитят сами, а клон на раннере sparse и
# depth=1 — `git rebase FETCH_HEAD` регулярно не сходится, и результат молча
# теряется. Contents API не зависит от состояния локального репозитория.
#
# Две ловушки, обе стоили потерянных прогонов:
#  1. при 404 `gh` печатает JSON ошибки в stdout — в sha попадал текст
#     {"message":"Not Found"}, и PUT всегда получал 409;
#  2. отчёт фолда ~144 КБ, это ~196 КБ base64. Через `-f content=...` такой
#     аргумент не влезает в командную строку: «Argument list too long», и
#     файл не загружался НИКОГДА. Поэтому тело запроса подаётся в stdin через
#     `--input -`, а base64 кладётся в JSON файлом.
#
#   bash tools/push_file58.sh "сообщение" файл1 [файл2 ...]
set -euo pipefail
message="${1:?нужно сообщение}"; shift || true
branch="${BRANCH:-arena/4a5c459f-smarty-chumkov}"
repo="${GITHUB_REPOSITORY:?нужен GITHUB_REPOSITORY}"
pushed=0
for path in "$@"; do
  [ -f "$path" ] || { echo "нет файла $path, пропуск"; continue; }
  ok=0
  for attempt in 1 2 3 4 5 6 7 8; do
    sha=$(gh api "repos/${repo}/contents/${path}?ref=${branch}" --jq '.sha' 2>/dev/null || true)
    case "$sha" in
      [0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]*) ;;
      *) sha="" ;;
    esac
    if [ -n "$sha" ] && [ "${#sha}" -ne 40 ]; then sha=""; fi
    # тело запроса собирается на диске: content туда пишет python, не shell
    body="/tmp/push58_body.json"
    python3 - "$path" "$message" "$branch" "${sha:-}" "$body" <<'PY'
import base64, json, sys
path, message, branch, sha, body = sys.argv[1:6]
payload = {
    "message": message,
    "content": base64.b64encode(open(path, "rb").read()).decode("ascii"),
    "branch": branch,
}
if sha:
    payload["sha"] = sha
open(body, "w", encoding="utf-8").write(json.dumps(payload))
PY
    if gh api -X PUT "repos/${repo}/contents/${path}" --input "$body" \
         --jq '.commit.sha' >/dev/null 2>&1; then ok=1; break; fi
    echo "попытка $attempt не прошла ($path), повтор через $((attempt * 2)) с"
    sleep $((attempt * 2))
  done
  rm -f /tmp/push58_body.json
  [ "$ok" = "1" ] || { echo "НЕ УДАЛОСЬ загрузить $path"; exit 1; }
  echo "загружено: $path ($(stat -c%s "$path") байт)"
  pushed=$((pushed+1))
done
echo "файлов загружено: $pushed"
