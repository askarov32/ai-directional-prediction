# Техническая документация MeshGraphNet Commercial

## 1. Назначение

Проект предназначен для построения AI-surrogate модели, которая заменяет длительные COMSOL расчёты быстрым прогнозом на графовой нейронной сети. Основной сценарий — распространение термоупругих волн в геологических средах при локальном нагреве, ударном/тепловом источнике или другом параметризованном физическом воздействии.

## 2. Физические поля

Модель может работать с любыми полями, экспортированными из COMSOL в формате `field (unit) @ t=value`. Типовой набор:

| Группа | Поля | Описание |
|---|---|---|
| Temperature | `T`, `temperature` | температура |
| Displacement | `u`, `v`, `w` | перемещения |
| Velocity | `ut`, `vt`, `wt` | скорости |
| Stress | `s11`, `s22`, `s33`, `s12`, `s13`, `s23` | компоненты напряжений |
| Strain | `exx`, `eyy`, `ezz`, `exy`, `exz`, `eyz` | компоненты деформаций |

## 3. Датасет

Каждый датасет — это один COMSOL-прогон или группа совместимых прогонов:

```text
datasets/<dataset_id>/
├── raw/
├── processed/
└── scenario.yaml
```

`scenario.yaml` нужен для conditional learning. Без него модель будет знать только поля, но не будет знать, при каких условиях они возникли.

## 4. Граф

COMSOL mesh преобразуется в граф:

- узлы графа = точки сетки;
- рёбра графа = связи из tetra/tri/edge элементов;
- если элементы не найдены, используется kNN fallback;
- edge features = `[dx, dy, dz, distance]`.

## 5. Вход модели

Для каждого узла формируется вектор:

```text
x_i = [node_static_i, material_features, scenario_features, dynamic_state_i(t)]
```

где:

```text
node_static_i = [x, y, z, distance_to_source, source_indicator]
material_features = [E, nu, rho, alpha, k, Cp]
scenario_features = [T_source, T_background, source_radius, source_center, dt, hashes]
dynamic_state_i(t) = [T, u, v, w, ut, vt, wt, stress, strain, ...]
```

## 6. Выход модели

Два режима:

```text
target_mode = delta:
  y = state(t+dt) - state(t)

target_mode = absolute:
  y = state(t+dt)
```

Для rollout рекомендуется `delta`, потому что он обычно устойчивее на последовательном прогнозе.

## 7. MeshGraphNet

Архитектура:

```text
Encoder(node, edge) → Processor(message passing × K) → Decoder
```

Message passing:

```text
e_ij' = MLP_e([h_i, h_j, e_ij]) + e_ij
m_i = Σ_j e_ji'
h_i' = MLP_v([h_i, m_i]) + h_i
```

## 8. Дообучение

Дообучение выполняется от checkpoint:

```bash
python scripts/finetune_model.py --checkpoint outputs/checkpoints/best_model.pt --dataset_ids new_rock_dataset
```

Рекомендуемые режимы:

| Случай | Что делать |
|---|---|
| новая температура той же породы | маленький LR, 30–80 эпох |
| новая порода, та же физика | LR 5e-5–1e-4, 80–150 эпох |
| новая физика/граничные условия | лучше добавить несколько COMSOL-прогонов и переобучить base/fine-tune |

## 9. Валидация

Оценка качества должна включать:

- RMSE/MAE по каждому полю;
- relative RMSE;
- ошибка максимальной температуры;
- ошибка максимального von Mises stress;
- стабильность autoregressive rollout;
- сравнение с COMSOL ground truth по времени.

## 10. Ограничения

1. Модель не заменяет COMSOL вне распределения обучения.
2. Для новой физики нужен хотя бы небольшой набор COMSOL-прогонов.
3. Если пользователь меняет граничные условия, их нужно кодировать в `scenario.yaml`.
4. Нельзя оценивать качество только по красивой анимации — нужна метрика против COMSOL.
