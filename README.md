# Финальный проект Physical AI

Выберите одного из двух роботов и одну из четырёх задач. Для выбранной пары
пройдите полный цикл RL → демонстрации → BC → робастность.
Начните с [задания](ASSIGNMENT.md). Справочник API: [MuJoCo](MUJOCO_GUIDE.md).

## Установка

Python 3.12, Linux. На Windows используйте WSL2. GPU необязательна для просмотра
и функциональных проверок; для больших BC-экспериментов рекомендуется GPU.
CPU-версия PPO использует процессы MuJoCo. Обучение эксперта занимает существенно
больше времени, чем короткая проверка установки; измеряйте скорость на своём
оборудовании. Время зависит от выбранной пары и необходимых промежуточных стадий
обучения. Сбор изображений и обучение BC требуют дополнительного времени.

Распакуйте студенческий архив и откройте терминал в его корне, либо клонируйте
выданный организаторами студенческий репозиторий.

```bash
sudo apt-get update
sudo apt-get install -y python3.12-venv python3.12-dev libosmesa6 libgl1 libglfw3 ffmpeg build-essential linux-libc-dev
python3.12 -m venv .venv
source .venv/bin/activate
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-lock.txt
export MUJOCO_GL=osmesa
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 LP_NUM_THREADS=2
python -m pytest -q
python -m physical_ai.scene_tools --robot ur5e --task cup_plate --snapshot runs/scene.png
```

Работайте из корня проекта: там находятся assets и конфигурации. Альтернатива
локальной установке — `./scripts/run_container.sh`; CPU Docker содержит системные
библиотеки, сохраняет результаты в смонтированный каталог и поддерживает snapshots
и видео без дисплея. Он не предоставляет графический рабочий стол.

Для интерактивного окна на Linux с графической сессией:

```bash
MUJOCO_GL=glfw python -m physical_ai.scene_tools --robot iiwa14 --task cup_distractor
```

Для сервера без дисплея используйте snapshots/видео. При рабочем NVIDIA-драйвере
можно выбрать `MUJOCO_GL=egl`. Для CUDA-обучения BC установите PyTorch 2.7.1
под свою CUDA-среду; поставляемый Docker использует CPU-сборку.

Сцены: `scenes/{ur5e,iiwa14}_{cup_plate,cup_shelf,cup_distractor,color_match}.xml`.
В команду просмотра подставляйте любую пару. Исходные роботы и лицензии в `assets/`.

## Проверка запуска обучения

```bash
python -m physical_ai.rl --robot ur5e --task cup_plate \
  --out runs/smoke_rl --total-steps 64 --n-steps 32 --n-envs 1
```

Это проверка обновления весов и сохранения, не обучение успешного эксперта.
Полные команды обучения, сбора и оценки находятся в `ASSIGNMENT.md`.
Начните с `configs/ppo_calibrated.json`: он задаёт проверенную последовательность
стадий PPO. Команда с `--target` обучает эксперта выбранной пары и необходимые
для него промежуточные модели. Остальные пары выполнять не требуется.
Промежуточные состояния curriculum — предоставленная помощь обучению; итоговый
SR измеряйте на обычных начальных состояниях.
Логи: `tensorboard --logdir runs`; метрики оценки сохраняются в JSON и CSV.

| Каталог/файл | Назначение |
|---|---|
| `physical_ai/env.py`, `scenes.py` | среда, управление, генерация сцен |
| `physical_ai/rl.py`, `curriculum.py`, `train_curriculum.py` | PPO, перенос между роботами, curriculum и воспроизведение рецепта |
| `physical_ai/data.py`, `lerobot_data.py` | сбор и чтение LeRobot v3 (Parquet + MP4) |
| `physical_ai/bc.py` | визуальный baseline, проприоцепция, экспорт |
| `physical_ai/evaluate.py` | запуск политики и подсчёт доли успешных эпизодов |
| `physical_ai/scene_tools.py` | просмотр и инспектор параметров |
| `configs/` | исходные конфигурации и формат редактирования |
| `REPORT_TEMPLATE.md` | структура отчёта |
| `submission.example.json` | модели для одной выбранной пары |

Данные и checkpoints не коммитятся автоматически. Сохраняйте их отдельно с
контрольными суммами. Источники и лицензии: `THIRD_PARTY_NOTICES.md`.
