import random
import sys

import numpy as np
import torch
from learn import (
    HIDDEN_SIZE,
    INPUT_SIZE,
    NUM_LAYERS,
    OUTPUT_SIZE,
    PREDICT_LEN,
    SEQ_LEN,
    CursorLSTM,
)
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

        self.button_A = ButtonWidget("Button", [100, 400], self.panel)
        self.button_A.button_press_signal.connect(self.recv_button_press)

        self.cursor_pos_list = []
        self.max_cursor_pos_list = 500
        self.realtime_cursor = CursorWidget(self.panel)
        self.delay_cursor_widgets = []
        for i, d in enumerate(range(50, 300, 100)):
            self.delay_cursor_widgets.append(CursorWidget(self.panel))
            self.delay_cursor_widgets[i].delay_index = d

        self.predicted_cursor = CursorWidget(self.panel)
        self.predicted_cursor.setStyleSheet(
            f"""
        QLabel{{
            background-color: rgba(255,0,0,100);
            min-width: %spx;
            min-height: %spx;
            max-width: %spx;
            max-height: %spx;
            border-radius: %spx;
        }}
        """
            % (
                self.predicted_cursor.cursor_size,
                self.predicted_cursor.cursor_size,
                self.predicted_cursor.cursor_size,
                self.predicted_cursor.cursor_size,
                self.predicted_cursor.cursor_size / 2,
            )
        )

        self.main_timer = QtCore.QTimer()
        self.main_timer.timeout.connect(self.update)
        self.main_timer.start(1)

        if not record:
            self.cursor_model = CursorLSTM(
                INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, NUM_LAYERS, 100
            )
            self.cursor_model.load_state_dict(
                torch.load(
                    "model/cursorlstm_ubuntu.pth",
                    map_location=torch.device("mps"),
                )
            )
            self.cursor_model.to(self.cursor_model.device)
            self.cursor_model.eval()

    def update(self):
        self.cursor_pos = self.mapFromGlobal(QtGui.QCursor.pos())

        self.relative_cursor_pos = [
            self.cursor_pos.x() / self.width(),
            self.cursor_pos.y() / self.height(),
        ]
        for i in self.relative_cursor_pos:
            if i < 0:
                self.relative_cursor_pos[self.relative_cursor_pos.index(i)] = 0
            elif i > 1:
                self.relative_cursor_pos[self.relative_cursor_pos.index(i)] = 1
        self.record_data = np.append(
            self.record_data,
            np.array(
                [[self.relative_cursor_pos[0], self.relative_cursor_pos[1], 0]]
            ),
            axis=0,
        )

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

        if not self.record:
            if len(self.record_data) >= SEQ_LEN:
                input_seq = np.array(self.record_data[-SEQ_LEN:][:, :2])
                input_seq = (
                    torch.tensor(input_seq, dtype=torch.float32)
                    .unsqueeze(0)
                    .to(self.cursor_model.device)
                )
                with torch.no_grad():
                    predictions = self.cursor_model(input_seq)
                    predicted_x = predictions[0, -1, 0].item() * self.width()
                    predicted_y = predictions[0, -1, 1].item() * self.height()
                    # predicted_click = predictions[0, -1, 2].item()
                    self.predicted_cursor.cursor_pos = [
                        predicted_x,
                        predicted_y,
                    ]

    def delay_cursor_update(self, delay_index):
        cursor_pos_list_len = len(self.cursor_pos_list)
        if cursor_pos_list_len < delay_index:
            return -cursor_pos_list_len
        else:
            return -delay_index

    def recv_button_press(self):
        self.record_data = np.append(
            self.record_data,
            np.array(
                [[self.relative_cursor_pos[0], self.relative_cursor_pos[1], 1]]
            ),
            axis=0,
        )

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
    window = MainWindow(record=False)
    window.show()
    sys.exit(app.exec())
