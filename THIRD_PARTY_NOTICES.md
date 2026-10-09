# Источники и лицензии

Новый код проекта распространяется под MIT. Модели роботов имеют собственные
лицензии BSD-3-Clause, сохранённые в `assets/*/LICENSE`; MIT на них не распространяется.
Исходные модели взяты из [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie).
Точная ревизия записана в `assets/MENAGERIE_REVISION`. Меши и исходные MJCF сохранены;
добавления захвата и окружения выполняет `physical_ai/scenes.py`.

Методическая преемственность:

- [Практика 1](https://github.com/Yandex-Practicum/ap-physicalai-1): PPO, обучение,
  checkpoints, эксперименты в симуляторе. Здесь PPO реализован через Stable Baselines3
  с обычным MuJoCo, вместо RSL-RL/MJX: одна физическая среда для обучения и сбора.
- [Практика 3](https://github.com/Yandex-Practicum/ap-physiclaai-3): визуомоторный BC,
  автоматический сбор демонстраций в LeRobot v3, отдельная rollout-оценка.
- [Практика 4](https://github.com/Yandex-Practicum/ap-physiclaai-4): добавление
  проприоцепции, сравнение baseline, абляции.

Код практик не скопирован. Готовый дистиллированный эксперт практик 3/4 не используется.
CNN-baseline здесь небольшой и обучается с нуля: обязательной загрузки сторонних
предобученных весов нет. Студент может заменить энкодер.

Происхождение исходных описаний: UR5e основан на URDF проекта
[ros-industrial/universal_robot](https://github.com/ros-industrial/universal_robot/tree/kinetic-devel/ur_e_description),
KUKA iiwa 14 — на описании разработчиков
[Drake](https://github.com/RobotLocomotion/drake/blob/master/manipulation/models/iiwa_description/urdf/iiwa14_spheres_dense_collision.urdf).
Лицензии и уведомления об авторских правах в `assets/*/LICENSE` сохранены.
