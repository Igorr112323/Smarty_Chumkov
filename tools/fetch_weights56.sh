#!/usr/bin/env bash
# Скачать веса одного фолда из прогона 44218112557 и распаковать в
# weights/ablation51/ten/fold<N>. Один фолд = 5 сидов, ~537 МБ архив,
# ~1,6 ГБ на диске. Архив удаляется сразу, чтобы не копить 5 ГБ.
set -euo pipefail

fold="${1:?нужен номер фолда}"
repo="${GITHUB_REPOSITORY:?нужен GITHUB_REPOSITORY}"
run="${SOURCE_RUN:-44218112557}"
target="weights/ablation51/ten/fold${fold}"

echo "=== фолд ${fold}: ищу артефакт weights56-outer${fold} в прогоне ${run}"
id=$(gh api "repos/${repo}/actions/runs/${run}/artifacts" --paginate \
      --jq ".artifacts[]|select(.name==\"weights56-outer${fold}\")|.id")
test -n "${id}" || { echo "ОШИБКА: артефакт весов для фолда ${fold} не найден"; exit 1; }
echo "artifact id: ${id}"

rm -f /tmp/w.zip
curl -sSL --retry 3 --retry-delay 5 \
  -H "Authorization: Bearer ${GH_TOKEN}" \
  -H "Accept: application/vnd.github+json" \
  "https://api.github.com/repos/${repo}/actions/artifacts/${id}/zip" -o /tmp/w.zip
echo "размер архива: $(stat -c%s /tmp/w.zip) байт"
unzip -t -q /tmp/w.zip >/dev/null || { echo "ОШИБКА: архив повреждён"; exit 1; }

rm -rf /tmp/unzip && mkdir -p /tmp/unzip
unzip -q -o /tmp/w.zip -d /tmp/unzip
rm -f /tmp/w.zip
echo "верхний уровень:"; ls /tmp/unzip | head -5

mkdir -p "${target}"
found=$(find /tmp/unzip -name manifest.json | head -1)
test -n "${found}" || { echo "ОШИБКА: в архиве нет manifest.json"; find /tmp/unzip -maxdepth 3 | head -20; exit 1; }
cp -r "$(dirname "${found}")/." "${target}/"
rm -rf /tmp/unzip

test -f "${target}/manifest.json" || { echo "ОШИБКА: manifest.json не на месте"; exit 1; }
echo "готово: ${target} ($(du -sh "${target}" | cut -f1))"
