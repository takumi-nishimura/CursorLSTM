import random
import sys

import numpy as np
import torch
from PySide6 import QtCore, QtGui, QtWidgets


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

        self.button_A = ButtonWidget("A", [100, 400], self.panel)
        self.button_A.button_press_signal.connect(self.recv_button_press)
        self.button_B = ButtonWidget("B", [350, 400], self.panel)
        self.button_B.button_press_signal.connect(self.recv_button_press)
        self.button_C = ButtonWidget("C", [600, 400], self.panel)
        self.button_C.button_press_signal.connect(self.recv_button_press)

        self.realtime_cursor = CursorWidget(self.panel)
        self.realtime_cursor.cursor_pos = [self.width() / 2, self.height() / 3]

        self.target_button = self.button_A
        self.cursor_trajectory = self.generate_cursor_trajectory()
        self.cursor_iter = iter(self.cursor_trajectory)

        self.main_timer = QtCore.QTimer()
        self.main_timer.timeout.connect(self.update)
        self.main_timer.start(1)

    def update(self):
        if self.cursor_iter is not None:
            try:
                next_cursor_pos = next(self.cursor_iter)
                self.realtime_cursor.cursor_pos = [
                    next_cursor_pos[0],
                    next_cursor_pos[1],
                ]
            except StopIteration:
                self.cursor_iter = None
                self.target_button.button_press_signal.emit()

    def delay_cursor_update(self, delay_index):
        cursor_pos_list_len = len(self.cursor_pos_list)
        if cursor_pos_list_len < delay_index:
            return -cursor_pos_list_len
        else:
            return -delay_index

    def recv_button_press(self):
        self.generate_button_pos()
        self.cursor_trajectory = self.generate_cursor_trajectory()
        self.cursor_iter = iter(self.cursor_trajectory)

    def generate_button_pos(self):
        for b in self.findChildren(ButtonWidget):
            while True:
                pos = [
                    random.randint(0, self.width() - 100),
                    random.randint(0, self.height() - 100),
                ]
                if not any(
                    button.geometry().intersects(
                        QtCore.QRect(*pos, 100 + 10, 100 + 10)
                    )
                    for button in self.findChildren(ButtonWidget)
                ):
                    b.move(*pos)
                    break

    def generate_cursor_trajectory(self):
        target = np.array(
            [self.target_button.x() + 50, self.target_button.y() + 50]
        )
        start = np.array(
            [
                self.realtime_cursor.cursor_pos[0],
                self.realtime_cursor.cursor_pos[1],
            ]
        )

        distance = np.linalg.norm(target - start)
        duration = 1 * distance

        t = np.linspace(0, duration, 500)
        tau = t / duration

        x = start[0] + (-target[0] + start[0]) * (
            15 * tau**4 - 6 * tau**5 - 10 * tau**3
        )
        y = start[1] + (-target[1] + start[1]) * (
            15 * tau**4 - 6 * tau**5 - 10 * tau**3
        )

        trajectory = np.vstack((x, y)).T
        return trajectory

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Q:
            self.close()

    def closeEvent(self, event):
        np.savetxt("data/record_data.csv", self.record_data, delimiter=",")
        event.accept()


class ButtonWidget(QtWidgets.QPushButton):
    button_press_signal = QtCore.Signal()

    def __init__(self, text, pos, parent=None):
        super().__init__(parent)
        self.parent_widget = parent
        self.button_size = 100
        self.setText(text)
        self.resize(self.button_size, self.button_size)
        self.move(pos[0], pos[1])

        self.clicked.connect(self.button_clicked)

    def button_clicked(self):
        self.button_press_signal.emit()


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
