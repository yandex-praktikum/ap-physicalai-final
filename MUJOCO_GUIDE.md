# MuJoCo: необходимые операции

Все примеры выполняются из корня репозитория. `model` — устройство мира и его
параметры; `data` — текущее динамическое состояние. Единицы: метры, секунды,
килограммы, радианы. Кватернионы MuJoCo записываются как **w, x, y, z**.

```python
import mujoco
import numpy as np
from physical_ai.env import ManipulationEnv

env = ManipulationEnv('ur5e', 'cup_plate')
env.reset(seed=0)
model, data = env.model, env.data
```

## Суставы и приводы

```python
for i in range(model.njnt):
    print(i, model.joint(i).name, model.jnt_type[i], model.jnt_range[i])

joint = model.joint('shoulder_pan_joint')
qadr = int(joint.qposadr[0])
vadr = int(joint.dofadr[0])
print(data.qpos[qadr], data.qvel[vadr])
for i in range(model.nu):
    print(i, model.actuator(i).name, model.actuator_ctrlrange[i])
```

Индекс сустава не равен индексу в `qpos` или `qvel`: свободное тело занимает
7 координат и 6 скоростей. У шарнира обычно по одной. Используйте адреса.

```python
# В этом проекте приводы руки позиционные; ctrl — целевые углы, не моменты.
motor = model.actuator('shoulder_pan').id
lo, hi = model.actuator_ctrlrange[motor]
data.ctrl[motor] = np.clip(-1.0, lo, hi)
mujoco.mj_step(model, data)
```

Один `mj_step` продвигает физику на `model.opt.timestep`. `env.step(action)`
выполняет полный управляющий шаг с IK и несколькими физическими подшагами.
Не смешивайте оба способа внутри тренировочного цикла.

## Положение и скорость объекта

```python
mujoco.mj_forward(model, data)
body = data.body('cup0')
world_position = body.xpos.copy()
world_quaternion = body.xquat.copy()
velocity = np.zeros(6)
mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY,
                        body.id, velocity, 0)
angular_velocity = velocity[:3]  # мировые оси
linear_velocity = velocity[3:]
```

`body_pos` в model — позиция относительно родителя, `xpos` в data — вычисленная
мировая позиция. `xpos` не является способом перемещения тела.

## Перемещение свободного тела при настройке/reset

```python
joint = model.joint('cup0_free')
qa, va = int(joint.qposadr[0]), int(joint.dofadr[0])
new_position = np.array([0.42, -0.14, 0.438])
data.qpos[qa:qa+3] = new_position
data.qpos[qa+3:qa+7] = [1, 0, 0, 0]
data.qvel[va:va+6] = 0
mujoco.mj_forward(model, data)
```

Это редактирование начального состояния. Использовать его для переноса предмета
политикой запрещено. `mj_forward` обновляет производные величины без шага времени.
Для тела без свободного сустава начальная локальная трансформация задаётся
`pos`/`quat` в MJCF; изменения структуры модели требуют её повторной загрузки.

## Физические и визуальные параметры

| Параметр MJCF | Где читать в model | Значение |
|---|---|---|
| `geom friction` | `geom_friction` | скольжение, кручение, качение |
| `geom mass` / `density` | итоговая `body_mass`, `body_inertia` | масса и инерция тела выводятся из геометрий |
| `joint damping` | `dof_damping` | вязкое сопротивление |
| `joint armature` | `dof_armature` | добавочная инерция привода |
| `geom rgba` | `geom_rgba` | цвет и прозрачность |
| `geom size` | `geom_size` | размеры, смысл зависит от типа геометрии |
| `option timestep` | `opt.timestep` | шаг интегрирования |

```python
geom = model.geom('table_top').id
model.geom_rgba[geom] = [.55, .52, .48, 1]
model.geom_friction[geom] = [.9, .01, .001]
mujoco.mj_forward(model, data)
```

Изменение размера, массы, положения статических элементов или инерции надёжнее
выполнять в MJCF и пересоздавать `MjModel`/`MjData`: одной записи `body_mass`
недостаточно для согласованной инерции и всех производных констант. Не меняйте
размеры геометрии во время rollout без явного экспериментального протокола.

## Параметры сцены без редактирования Python

```bash
python -m physical_ai.scene_tools --robot ur5e --task cup_plate --list-parameters geom
python -m physical_ai.scene_tools --robot ur5e --task cup_plate --list-parameters body
python -m physical_ai.scene_tools --robot ur5e --task cup_plate --list-parameters camera
python -m physical_ai.scene_tools --robot ur5e --task cup_plate --list-parameters light
python -m physical_ai.scene_tools --parameters configs/scene_example.json --snapshot runs/modified.png
```

Формат JSON: `{секция: {имя: {атрибут: [значения]}}}`. Поддерживаются:
`body: pos, quat`; `geom: rgba, friction, mass, size, pos, quat`;
`light: pos, dir, diffuse, ambient`; `camera: pos, quat, fovy`.
Неизвестное имя или атрибут вызывает ошибку. Более сложные изменения выполняются
в генераторе `physical_ai/scenes.py`. Перед экспериментом просмотрите результат,
проверьте отсутствие пересечений и сохранение смысла задачи.

Сохраняйте JSON рядом с результатами. `--parameters FILE` применяется к просмотру,
RL и оценке; `--scene-bank FILE` у сборщика принимает список таких конфигураций.
После редактирования генератора обновите XML командой `python -m physical_ai.scenes`.

```python
env.close()  # освободить контекст рендера
```
