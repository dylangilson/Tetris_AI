"""
* Dylan Gilson
* dylan.gilson@outlook.com
* March 1, 2023
"""

from keras.callbacks import TensorBoard
from tensorflow.summary import FileWriter


class CustomTensorBoard(TensorBoard):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.writer = FileWriter(self.log_dir)

    def log(self, step, **stats):
        self._write_logs(stats, step)
