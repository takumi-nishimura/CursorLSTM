import sys
import threading

from env import TaskEnv
from logger import logger
from PySide6 import QtCore, QtGui, QtWidgets


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.env = TaskEnv()
        threading.Thread(target=self.env.loop, daemon=True).start()

        self.setWindowTitle("View")
        self.setGeometry(
            self.env.main_geometry.x,
            self.env.main_geometry.y,
            self.env.main_geometry.width,
            self.env.main_geometry.height,
        )

        self.panel = QtWidgets.QWidget()
        self.setCentralWidget(self.panel)

        self.button_state_list = self.env.getButtonStates()
        self.button_widgets = [
            ButtonWidget(
                name=button_state.name,
                pos=(button_state.x, button_state.y),
                size=(button_state.width, button_state.height),
                parent=self.panel,
            )
            for button_state in self.button_state_list
        ]

        self.agent_cursor_widget = CursorWidget(
            name=self.env.agent_cursor.name,
            pos=(
                self.env.agent_cursor.center_x,
                self.env.agent_cursor.center_y,
            ),
            color=(0, 0, 255),
            parent=self.panel,
        )
        self.operator_cursor_widget = CursorWidget(
            name=self.env.operator_cursor.name,
            pos=(
                self.env.operator_cursor.center_x,
                self.env.operator_cursor.center_y,
            ),
            color=(255, 0, 0),
            parent=self.panel,
        )

        self.main_timer = QtCore.QTimer()
        self.main_timer.timeout.connect(self.update)
        self.main_timer.start(1)

    def update(self):
        for button in self.button_widgets:
            _state = self.check_pair(button)
            button.move(_state.x, _state.y)

        self.agent_cursor_widget.cursorMove(
            self.env.agent_cursor.center_x, self.env.agent_cursor.center_y
        )

    def check_pair(self, widget):
        self.button_state_list = self.env.getButtonStates()
        for state in self.button_state_list:
            if widget.text() == state.name:
                return state

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Q:
            self.close()


class ButtonWidget(QtWidgets.QPushButton):
    def __init__(self, name, pos, size, parent):
        super().__init__(parent)

        self.setText(name)
        self.move(pos[0], pos[1])
        self.resize(size[0], size[1])
        self.setCheckable(True)


class CursorWidget(QtWidgets.QLabel):
    def __init__(self, name, pos, color, parent):
        super().__init__(parent)
        self.name = name
        self.center_pos = pos
        self.color = color
        self.cursor_size = 20
        self.resize(self.cursor_size, self.cursor_size)
        self.setStyleSheet(
            f"""
        QLabel{{
            background-color: rgba(%s,%s,%s,70);
            min-width: %spx;
            min-height: %spx;
            max-width: %spx;
            max-height: %spx;
            border-radius: %spx;
        }}
        """
            % (
                self.color[0],
                self.color[1],
                self.color[2],
                self.cursor_size,
                self.cursor_size,
                self.cursor_size,
                self.cursor_size,
                self.cursor_size / 2,
            )
        )
        self.cursorMove(pos[0], pos[1])

    def cursorMove(self, x, y):
        self.move(
            x - self.cursor_size / 2,
            y - self.cursor_size / 2,
        )


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
