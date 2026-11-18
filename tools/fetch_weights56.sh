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
echo "верхний уровень архива:"; ls /tmp/unzip | head -5

# архив собран из весов/ablation51/ten/fold<N>/, то есть внутри уже есть
# либо каталог fold<N>, либо сразу seed_0 … seed_4. Копируется тот каталог,
# в котором лежат сиды; брать первый попавшийся manifest.json нельзя —
# первый по алфавиту это seed_0/manifest.json, и тогда копируется один сид.
source_dir=""
if [ -d "/tmp/unzip/fold${fold}/seed_0" ]; then
  source_dir="/tmp/unzip/fold${fold}"
elif [ -d "/tmp/unzip/seed_0" ]; then
  source_dir="/tmp/unzip"
else
  found=$(find /tmp/unzip -maxdepth 3 -type d -name 'seed_*' | head -1)
  test -n "${found}" || { echo "ОШИБКА: в архиве нет каталогов seed_*"; find /tmp/unzip -maxdepth 3 | head -20; exit 1; }
  source_dir=$(dirname "${found}")
fi
echo "каталог с сидами: ${source_dir}"
echo "сидов в архиве: $(find "${source_dir}" -maxdepth 1 -type d -name 'seed_*' | wc -l)"

rm -rf "${target}" && mkdir -p "${target}"
cp -r "${source_dir}"/. "${target}/"
rm -rf /tmp/unzip

count=$(find "${target}" -maxdepth 1 -type d -name 'seed_*' | wc -l)
echo "сидов на месте: ${count} ($(du -sh "${target}" | cut -f1))"
test "${count}" -eq 5 || { echo "ОШИБКА: ожидалось 5 сидов, на месте ${count}"; ls "${target}"; exit 1; }
for seed_dir in "${target}"/seed_*; do
  test -f "${seed_dir}/manifest.json" || { echo "ОШИБКА: нет манифеста в ${seed_dir}"; exit 1; }
done
echo "готово: ${target}"
