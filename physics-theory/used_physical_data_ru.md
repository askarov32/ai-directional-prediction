# Какие физические данные используются

## 1. Физические данные каталога геологических сред

Каждая геологическая среда в каталоге `backend/data/media/catalog.json` содержит набор физических свойств.

Основные поля:

- `rho` — плотность, кг/м³
- `porosity_total` — общая пористость
- `porosity_effective` — эффективная пористость
- `vp` — скорость продольной волны `P-wave`
- `vs` — скорость поперечной волны `S-wave`
- `thermal_conductivity` — теплопроводность
- `heat_capacity` — теплоемкость
- `thermal_expansion` — коэффициент линейного теплового расширения

Также для каждой среды заданы допустимые диапазоны:

- `temperature_c`
- `pressure_mpa`

## 2. Какие данные реально используются в текущем PINN inference

В текущем `pinn-service` реально участвуют следующие физические величины:

### Из свойств среды

- `rho`
- `vp`
- `vs`
- `thermal_conductivity`
- `heat_capacity`
- `thermal_expansion`
- `porosity_effective`

### Из сценария

- `temperature_c`
- `pressure_mpa`
- `time_ms`

### Из источника

- `source.type`
- `source.x`, `source.y`, `source.z`
- `source.amplitude`
- `source.frequency_hz`
- `source.direction = [dx, dy, dz]`

### Из точки наблюдения

- `probe.x`, `probe.y`, `probe.z`

### Из области расчета

- `domain.type` (`rect_2d` или `rect_3d`)
- `domain.size.lx`, `domain.size.ly`, `domain.size.lz`
- `domain.resolution.nx`, `domain.resolution.ny`, `domain.resolution.nz`

## 3. Какие признаки реально подаются в модель

Текущая baseline-модель использует исторический набор признаков:

- `x`
- `y`
- `z`
- `t`
- `youngs_modulus`
- `poissons_ratio`
- `density`
- `thermal_expansion`
- `thermal_conductivity`
- `heat_capacity`

То есть на вход сети идут:

- координаты точки;
- время;
- упругие свойства;
- плотность;
- тепловые свойства.

## 4. Какие дополнительные физические признаки уже поддерживаются кодом

В текущем коде `build_feature_vector(...)` подготовлены и дополнительные признаки, которые можно использовать в следующих checkpoint’ах:

- `temperature_c`
- `pressure_mpa`
- `source_x`, `source_y`, `source_z`
- `source_amplitude`
- `source_frequency_hz`
- `source_dir_x`, `source_dir_y`, `source_dir_z`
- `source_probe_dx`, `source_probe_dy`, `source_probe_dz`
- `source_probe_distance`
- `domain_lx`, `domain_ly`, `domain_lz`
- `domain_nx`, `domain_ny`, `domain_nz`

Это важно:
- в коде они уже есть;
- но конкретно используемый checkpoint может быть обучен только на старом базовом наборе из 10 признаков.

## 5. Какие физические данные были в тренировочном COMSOL-датасете

Из ваших CSV были извлечены такие физические поля.

### Материальные свойства

- `E` — модуль Юнга
- `nu` — коэффициент Пуассона
- `rho` — плотность
- `alpha` — коэффициент теплового расширения

### Тепловые данные

- `T` — температура
- `k` — теплопроводность
- `Cp` — теплоемкость

### Кинематика

- `u`, `v`, `w` — компоненты смещения
- `ut`, `vt`, `wt` — компоненты скорости

### Напряжения

- `von Mises`
- `sx`, `sy`, `sz`
- `sxy`, `syz`, `sxz`

### Деформации

- `eX`, `eY`, `eZ`

### Геометрия

- `x`, `y`, `z`
- временная сетка `t`

## 6. Что использовалось в обучении baseline PINN

В baseline-тренировке основной выход модели:

- `T`
- `u`
- `v`
- `w`

Дополнительно в loss использовались:

- скорости `ut`, `vt`, `wt`
- тепловой residual

То есть обучение было гибридным:

- **supervised loss** по данным COMSOL;
- **velocity consistency loss**;
- **thermal PDE residual loss**.

## 7. Какие величины используются для финального предсказания на фронт

После работы сети финальный backend/frontend response содержит:

- `direction_vector`
- `azimuth_deg`
- `elevation_deg`
- `magnitude`
- `wave_type`
- `travel_time_ms`
- `max_displacement`
- `max_temperature_perturbation`

Эти величины формируются не напрямую одним численным решателем, а через:

- выход нейросети;
- геометрию `source -> probe`;
- свойства среды;
- тепловые и волновые параметры;
- физически осмысленную постобработку.

## 8. Что особенно важно для объяснения на защите

Если коротко, то в проекте используются три уровня физических данных:

1. **Параметры породы**
   - плотность;
   - волновые скорости;
   - пористость;
   - тепловые свойства.

2. **Параметры сценария**
   - температура;
   - давление;
   - время;
   - характеристики источника.

3. **Пространственные данные**
   - координаты источника;
   - координаты зонда;
   - размеры домена;
   - 2D/3D конфигурация.

Именно комбинация этих данных позволяет получать directional prediction, а не просто один “средний” ответ.
