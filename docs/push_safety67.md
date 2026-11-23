# Аудит push-триггеров перед сдачей

Область: все **33** файла `.github/workflows/*.yml` опубликованного дерева ветки `arena/4a5c459f-smarty-chumkov`. Проверено YAML-структурой без запуска workflow, обучения, оценки или чтения Dataset. Для push в эту ветку на проверенном дереве:

- **13 workflow не имеют `on: push`**: `ci`, `cli52`, `evaluate`, `memory66`, `preflight65`, `final58`, `final59`, `final59c`, `infer56`, `prompt64`, `rerun56`, `seg63`, `seg63_corrected`.
- **19 workflow** сохраняют существующий фильтр `branches: [arena/5afadcb0-smarty-chumkov]`, который исключает текущую ветку: `ablation46`, `ablation47`, `ablation50`, `ablation51_dual`, `ablation51_ten`, `collect_base`, `diagnose_base34`, `final47_verify`, `inner43_audit`, `nested36`, `nested37`, `nested37_verify`, `nested38`, `nested44`, `predictions40`, `preflight44`, `reproduce45`, `speed51`, `train_base`.
- **1 workflow**, `ablation52_chunks`, ранее без branch-фильтра, теперь явно исключает текущую ветку через `branches-ignore`; его существующий path-фильтр и возможность push на другие ветки оставлены.

В восьми workflow (`final58`, `final59`, `final59c`, `infer56`, `prompt64`, `rerun56`, `seg63`, `seg63_corrected`) удалён `push`, который был ограничен только текущей веткой. `workflow_dispatch` и определения заданий не изменены. Остальные файлы сохранены без изменений. Проверка синтаксиса и сравнение разобранной YAML-структуры подтверждают, что в изменённых workflow остальные события и jobs не менялись.

**Границы гарантии:** это аудит текущего коммита и этой ветки, не запрет ручного запуска. `workflow_dispatch` и `ci` на `pull_request` остаются отдельными событиями; вручную их не запускали. Другие ветки могут по-прежнему запускать свои workflow. Позднейшее добавление/изменение файла workflow или объединение изменений из другой ветки требует повторного аудита.

Ранее случайно запущенный `cli52` и неопределённость чтения test 301 документированы отдельно в `docs/incident67_cli52.md`; результаты инцидента не использовались.
