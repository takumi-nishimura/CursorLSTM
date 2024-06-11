import random
import sys

import numpy as np
from PySide6 import QtCore, QtWidgets


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, record):
        super().__init__()
        self.setWindowTitle("Cursor LSTM")
        self.setGeometry(0, 0, 800, 600)
        self.setMouseTracking(True)

        self.record = record
        self.record_data = np.empty((0, 3))

        self.panel = QtWidgets.QWidget()
        self.setCentralWidget(self.panel)

        self.button_A = ButtonWidget(
            text="A", pos=[100, 400], width=50, height=50, parent=self.panel
        )
        self.button_A.button_press_signal.connect(self.recv_button_press)
        self.button_B = ButtonWidget(
            text="B", pos=[350, 400], width=150, height=150, parent=self.panel
        )
        self.button_B.button_press_signal.connect(self.recv_button_press)
        self.button_C = ButtonWidget(
            text="C", pos=[600, 400], parent=self.panel
        )
        self.button_C.button_press_signal.connect(self.recv_button_press)

        self.target_buttons = [self.button_A, self.button_B]

        self.realtime_cursor = CursorWidget(self.panel)
        self.realtime_cursor.cursor_pos = [self.width() / 2, self.height() / 3]

        self.target_button = self.button_A
        self.current_target_button = self.target_button
        self.cursor_trajectory = self.generate_cursor_trajectory()
        self.cursor_iter = enumerate(iter(self.cursor_trajectory), start=1)

        self.main_timer = QtCore.QTimer()
        self.main_timer.timeout.connect(self.update)
        self.main_timer.start(1)

    def update(self):
        if self.cursor_iter is not None:
            try:
                i, next_cursor_pos = next(self.cursor_iter)
                self.realtime_cursor.cursor_pos = [
                    next_cursor_pos[0],
                    next_cursor_pos[1],
                ]

            except StopIteration:
                self.cursor_iter = None
                if (
                    self.calc_overlap_area(
                        self.current_target_button, self.realtime_cursor
                    )
                    > 0
                ):
                    self.current_target_button.button_press_signal.emit(
                        self.current_target_button
                    )

    def calc_overlap_area(self, button, cursor):
        button_area = button.geometry()
        button_x1, button_y1 = button_area.x(), button_area.y()
        button_x2, button_y2 = (
            button_x1 + button_area.width(),
            button_y1 + button_area.height(),
        )

        cursor_area = cursor.geometry()
        cursor_x1, cursor_y1 = cursor_area.x(), cursor_area.y()
        cursor_x2, cursor_y2 = (
            cursor_x1 + cursor_area.width(),
            cursor_y1 + cursor_area.height(),
        )

        overlap_x1 = max(button_x1, cursor_x1)
        overlap_y1 = max(button_y1, cursor_y1)
        overlap_x2 = min(button_x2, cursor_x2)
        overlap_y2 = min(button_y2, cursor_y2)

        if overlap_x1 < overlap_x2 and overlap_y1 < overlap_y2:
            return (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
        else:
            return 0

    def recv_button_press(self, pressed_button):
        pressed_button.setChecked(True)

        if all(button.isChecked() for button in self.target_buttons):
            self.generate_button_pos()
            self.cursor_trajectory = self.generate_cursor_trajectory()
            self.cursor_iter = enumerate(iter(self.cursor_trajectory), start=1)
            [
                button.setChecked(False)
                for button in self.findChildren(ButtonWidget)
            ]

    def generate_button_pos(self):
        for b in self.findChildren(ButtonWidget):
            while True:
                pos = [
                    random.randint(0, self.width() - b.width()),
                    random.randint(0, self.height() - b.height()),
                ]
                if not any(
                    button.geometry().intersects(
                        QtCore.QRect(
                            *pos,
                            b.width(),
                            b.height(),
                        )
                    )
                    for button in self.findChildren(ButtonWidget)
                ):
                    b.move(*pos)
                    break

    def generate_cursor_trajectory(self):
        self.current_target_button = self.target_button
        other_buttons = [
            button
            for button in self.target_buttons
            if button != self.current_target_button
        ]

        start_pos = np.array(
            [
                self.realtime_cursor.cursor_pos[0],
                self.realtime_cursor.cursor_pos[1],
            ]
        )
        target_pos = np.array(
            [
                self.current_target_button.x()
                + self.current_target_button.width() / 2,
                self.current_target_button.y()
                + self.current_target_button.height() / 2,
            ]
        )
        others_pos = [
            np.array(
                [
                    button.x() + button.width() / 2,
                    button.y() + button.height() / 2,
                ]
            )
            for button in other_buttons
        ]

        target_distance = np.linalg.norm(target_pos - start_pos)
        others_distance = [
            np.linalg.norm(other_pos - start_pos) for other_pos in others_pos
        ]

        minimum_distance_index = np.argmin(others_distance)
        if others_distance[minimum_distance_index] < target_distance:
            if random.random() < 0.8:
                self.current_target_button = other_buttons[
                    minimum_distance_index
                ]

        return self.generate_minimal_trajectory(self.current_target_button)

    def change_target_button(self):
        self.cursor_trajectory = self.generate_minimal_trajectory(
            self.current_target_button
        )
        self.cursor_iter = enumerate(iter(self.cursor_trajectory), start=1)

    def generate_minimal_trajectory(self, target_button):
        start = np.array(
            [
                self.realtime_cursor.cursor_pos[0],
                self.realtime_cursor.cursor_pos[1],
            ]
        )
        target = np.array(
            [
                target_button.x() + target_button.width() / 2,
                target_button.y() + target_button.height() / 2,
            ]
        )

        distance = np.linalg.norm(target - start)
        duration = 1 * distance

        t = np.linspace(0, duration, 800)
        tau = t / duration

        x = start[0] + (-target[0] + start[0]) * (
            15 * tau**4 - 6 * tau**5 - 10 * tau**3
        )
        y = start[1] + (-target[1] + start[1]) * (
            15 * tau**4 - 6 * tau**5 - 10 * tau**3
        )

        return np.vstack((x, y)).T

    def delay_cursor_update(self, delay_index):
        cursor_pos_list_len = len(self.cursor_pos_list)
        if cursor_pos_list_len < delay_index:
            return -cursor_pos_list_len
        else:
            return -delay_index

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Q:
            self.close()

    def closeEvent(self, event):
        np.savetxt("data/record_data.csv", self.record_data, delimiter=",")
        event.accept()


class ButtonWidget(QtWidgets.QPushButton):
    button_press_signal = QtCore.Signal(QtWidgets.QWidget)

    def __init__(self, text, pos, width=100, height=100, parent=None):
        super().__init__(parent)
        self.parent_widget = parent
        self.button_width = width
        self.button_height = height
        self.setText(text)
        self.resize(self.button_width, self.button_height)
        self.move(pos[0], pos[1])
        self.setCheckable(True)

        self.clicked.connect(self.button_clicked)

    def button_clicked(self):
        self.button_press_signal.emit(self)


class CursorWidget(QtWidgets.QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, True)

        self.cursor_pos = [0, 0]
        self.delay_index = 0
        self.cursor_size = 20
        self.cursor_style = f"""
        QLabel{{
            background-color: rgba(83,191,245,100);
            min-width: %spx;
            min-height: %spx;
            max-width: %spx;
            max-height: %spx;
            border-radius: %spx;
        }}
        """ % (
            self.cursor_size,
            self.cursor_size,
            self.cursor_size,
            self.cursor_size,
            self.cursor_size / 2,
        )
        self.setStyleSheet(self.cursor_style)
        self.move(50, 50)

        self.cursor_update = QtCore.QTimer()
        self.cursor_update.timeout.connect(self.update_cursor)
        self.cursor_update.start(1)

    def update_cursor(self):
        self.move(
            self.cursor_pos[0] - self.cursor_size / 2,
            self.cursor_pos[1] - self.cursor_size / 2,
        )


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow(record=False)
    window.show()
    sys.exit(app.exec())
