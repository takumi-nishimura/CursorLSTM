import random
import sys

from PySide6 import QtCore, QtGui, QtWidgets


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cursor LSTM")
        self.setGeometry(0, 0, 800, 600)
        self.setMouseTracking(True)

        self.panel = QtWidgets.QWidget()
        self.setCentralWidget(self.panel)

        self.button_A = ButtonWidget("Button", [100, 400], self.panel)

        self.cursor_pos_list = []
        self.max_cursor_pos_list = 500
        self.realtime_cursor = CursorWidget(self.panel)
        self.delay_cursor_widgets = []
        for i, d in enumerate(range(50, 300, 100)):
            self.delay_cursor_widgets.append(CursorWidget(self.panel))
            self.delay_cursor_widgets[i].delay_index = d

        self.main_timer = QtCore.QTimer()
        self.main_timer.timeout.connect(self.update)
        self.main_timer.start(1)

    def update(self):
        self.cursor_pos = self.mapFromGlobal(QtGui.QCursor.pos())
        self.cursor_pos_list.append([self.cursor_pos.x(), self.cursor_pos.y()])
        if len(self.cursor_pos_list) > self.max_cursor_pos_list:
            self.cursor_pos_list.pop(0)

        self.realtime_cursor.cursor_pos = [
            self.cursor_pos_list[-1][0],
            self.cursor_pos_list[-1][1],
        ]

        for i in range(len(self.delay_cursor_widgets)):
            self.delay_cursor_widgets[i].cursor_pos = [
                self.cursor_pos_list[
                    self.delay_cursor_update(
                        self.delay_cursor_widgets[i].delay_index
                    )
                ][0],
                self.cursor_pos_list[
                    self.delay_cursor_update(
                        self.delay_cursor_widgets[i].delay_index
                    )
                ][1],
            ]

    def delay_cursor_update(self, delay_index):
        cursor_pos_list_len = len(self.cursor_pos_list)
        if cursor_pos_list_len < delay_index:
            return -cursor_pos_list_len
        else:
            return -delay_index


class ButtonWidget(QtWidgets.QPushButton):
    def __init__(self, text, pos, parent=None):
        super().__init__(parent)
        self.button_size = 100
        self.setText(text)
        self.resize(self.button_size, self.button_size)
        self.move(pos[0], pos[1])

        self.clicked.connect(self.button_clicked)

    def button_clicked(self):
        next_button_pos_x = random.randint(
            0, self.parentWidget().width() - self.button_size
        )
        next_button_pos_y = random.randint(
            0, self.parentWidget().height() - self.button_size
        )

        self.move(next_button_pos_x, next_button_pos_y)


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
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
