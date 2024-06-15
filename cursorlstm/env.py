import gc
import random
import time
import weakref

import numpy as np
from logger import logger


class TaskEnv:
    def __init__(self):
        self.env_fps = 120

        self.main_geometry = MainGeometry(0, 0, 800, 600)

        self.button_A = ButtonState(name="A", size=(50, 50))
        self.button_B = ButtonState(name="B", size=(300, 300))
        self.button_C = ButtonState(name="C", size=(100, 100))

        self.agent_cursor = CursorState(
            name="agent",
            pos=(self.main_geometry.width / 2, self.main_geometry.height / 3),
        )
        self.operator_cursor = CursorState(name="operator", pos=(50, 50))

        self.target_buttons = [self.button_A, self.button_B]
        self.agent_cursor.target_button = self.button_A

        self.init_env()

    def init_env(self):
        self.change_button_pos()
        self.agent_cursor.current_target_button = (
            self.agent_cursor.target_button
        )

        self.agent_cursor.trajectory = self.plan_cursor_target(
            self.agent_cursor
        )
        self.agent_cursor.trajectory_iter = self.make_iter(
            self.agent_cursor.trajectory
        )

    def step(self):
        logger.info(
            f"{self.agent_cursor.center_x},{self.agent_cursor.center_y}"
        )

        if self.agent_cursor.trajectory_iter is not None:
            try:
                idx, pos = next(self.agent_cursor.trajectory_iter)
                self.agent_cursor.setPos(pos)
            except StopIteration:
                self.agent_cursor.trajectory_iter = None
                self.agent_cursor.setClick(True)

        self.agent_cursor.setClick(False)

    def loop(self):
        while True:
            try:
                _start = time.perf_counter()

                self.step()

                _run_time = time.perf_counter() - _start
                if _run_time < 1 / self.env_fps:
                    time.sleep(1 / self.env_fps - _run_time)

            except Exception as e:
                logger.error(e)
                break

    def getButtonStates(self):
        gc.collect()
        return ButtonState.get_instances()

    def change_button_pos(self):
        button_states = self.getButtonStates()
        for i, button in enumerate(button_states):
            button.setChecked(False)
            without_self_states = [
                state for state in button_states if state != button
            ]
            while True:
                pos = (
                    random.randint(0, self.main_geometry.width - button.width),
                    random.randint(
                        0, self.main_geometry.height - button.height
                    ),
                )

                if not any(
                    self.calc_overlap_area(
                        ButtonState(
                            "temporary", pos, (button.width, button.height)
                        ),
                        _button,
                    )
                    for _button in without_self_states
                ):
                    button.setPos(pos)
                    break

    def calc_overlap_area(self, widget_1, widget_2):
        overlap_x1 = max(widget_1.x, widget_2.x)
        overlap_y1 = max(widget_1.y, widget_2.y)
        overlap_x2 = min(
            widget_1.x + widget_1.width, widget_2.x + widget_2.height
        )
        overlap_y2 = min(
            widget_1.y + widget_1.height, widget_2.y + widget_2.height
        )

        if overlap_x1 < overlap_x2 and overlap_y1 < overlap_y2:
            return (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
        return 0

    def plan_cursor_target(
        self, cursor, change_probability=0.3, mode="minimum_jerk"
    ):
        other_buttons = [
            button
            for button in self.target_buttons
            if button != cursor.current_target_button
            and not button.isChecked()
        ]
        start_pos = np.array([cursor.center_x, cursor.center_y])
        target_pos = np.array(
            [
                cursor.current_target_button.x
                + cursor.current_target_button.width / 2,
                cursor.current_target_button.y
                + cursor.current_target_button.height / 2,
            ]
        )
        others_pos = [
            np.array(
                [button.x + button.width / 2, button.y + button.height / 2]
            )
            for button in other_buttons
        ]

        target_distance = np.linalg.norm(target_pos - start_pos)
        others_distance = [
            np.linalg.norm(other_pos - start_pos) for other_pos in others_pos
        ]

        if len(others_distance) > 0:
            min_distance_idx = np.argmin(others_distance)
            if others_distance[min_distance_idx] < target_distance:
                if random.random() < change_probability:
                    cursor.current_target_button = other_buttons[
                        min_distance_idx
                    ]

        return self.generate_trajectory(cursor, mode)

    def generate_trajectory(self, cursor, mode="minimum_jerk"):
        start = np.array(
            [
                cursor.center_x,
                cursor.center_y,
            ]
        )
        target = np.array(
            [
                cursor.current_target_button.x
                + cursor.current_target_button.width / 2,
                cursor.current_target_button.y
                + cursor.current_target_button.height / 2,
            ]
        )

        error = np.random.normal(0, 10, target.shape)
        target += error

        duration = np.linalg.norm(target - start)

        t = np.linspace(0, duration, 100)
        tau = t / duration

        if mode == "bezier":
            x = start[0] + (target[0] - start[0]) * (
                10 * tau**3 - 15 * tau**4 + 6 * tau**5
            )
            y = start[1] + (target[1] - start[1]) * (
                10 * tau**3 - 15 * tau**4 + 6 * tau**5
            )
        else:
            x = start[0] + (-target[0] + start[0]) * (
                15 * tau**4 - 6 * tau**5 - 10 * tau**3
            )
            y = start[1] + (-target[1] + start[1]) * (
                15 * tau**4 - 6 * tau**5 - 10 * tau**3
            )

        return np.vstack((x, y)).T

    def make_iter(self, trajectory):
        return enumerate(iter(trajectory), start=1)


class MainGeometry:
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height


class ButtonInstanceTracker(type):
    def __init__(cls, name, bases, attrs):
        super().__init__(name, bases, attrs)
        cls._instances = weakref.WeakSet()

    def __call__(cls, *args, **kwargs):
        instance = super().__call__(*args, **kwargs)
        if instance.name not in ["temporary"]:
            cls._instances.add(instance)
        return instance

    def get_instances(cls):
        return list(cls._instances)


class ButtonState(metaclass=ButtonInstanceTracker):
    def __init__(
        self, name: str, pos: tuple = (0, 0), size: tuple = (100, 100)
    ):
        self.name = name
        self.x = pos[0]
        self.y = pos[1]
        self.width = size[0]
        self.height = size[1]

        self.checked = False

    def setChecked(self, check: bool):
        self.checked = check

    def setPos(self, pos: tuple):
        self.x = pos[0]
        self.y = pos[1]

    def isChecked(self):
        return self.checked


class CursorState:
    def __init__(self, name, pos: tuple = (0, 0)):
        self.name = name
        self.size = 20
        self.center_x = pos[0]
        self.center_y = pos[1]
        self.click = False

        self.target_button = None
        self.current_target_button = None
        self.target_change = False
        self.trajectory = None
        self.trajectory_iter = None

    def setPos(self, pos: tuple):
        self.center_x = pos[0]
        self.center_y = pos[1]

    def setClick(self, click: bool):
        self.click = click


if __name__ == "__main__":
    env = TaskEnv()
    env.loop()
