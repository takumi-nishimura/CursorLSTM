import json
import logging
import logging.handlers
from datetime import datetime

import pandas as pd
from pythonjsonlogger import jsonlogger

DEBUG = logging.DEBUG
ERROR = logging.ERROR
INFO = logging.INFO
WARN = logging.WARN
WARNING = logging.WARNING

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

stream_formatter = logging.Formatter(
    "%(asctime)s.%(msecs)03d [%(levelname)-8s] %(module)s:%(lineno)d >> %(message)s",
    datefmt="%H:%M:%S",
)


streamHandler = logging.StreamHandler()
streamHandler.setFormatter(stream_formatter)
logger.addHandler(streamHandler)

json_formatter = jsonlogger.JsonFormatter(
    fmt="%(asctime)s %(module)s %(lineno)d  %(message)s",
    json_ensure_ascii=False,
)


def set_stream_filter(filt=[]):
    streamHandler.addFilter(LogTextFilter(filtText=filt))


def set_file_handler(filename, fileLogLevel, mode, filt=[]):
    __fileHandler = logging.FileHandler(filename=filename, mode=mode)
    __fileHandler.setFormatter(json_formatter)
    __fileHandler.setLevel(fileLogLevel)
    logger.addHandler(__fileHandler)
    if filt:
        __fileHandler.addFilter(LogTextFilter(filtText=filt))


class LogTextFilter(logging.Filter):
    def __init__(self, filtText):
        self.filt_text = filtText

    def filter(self, record):
        results = [True for r in self.filt_text if r in record.getMessage()]
        if results:
            return True
        return False


def read_log(path):
    d = []
    with open(path) as f:
        lines = f.readlines()
        for _, line in enumerate(lines):
            line_data = json.loads(line)
            d.append(line_data)
    df = pd.DataFrame(data=d)
    df["asctime"] = df["asctime"].apply(
        lambda x: (
            datetime.strptime(x, "%Y-%m-%d %H:%M:%S,%f")
            if x is not None
            else None
        )
    )
    return df
