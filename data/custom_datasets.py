import pandas as pd
import os
from pathlib import Path
import sys
from sklearn.preprocessing import MinMaxScaler
import numpy as np

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.abspath(ROOT))  # relative
WORK_DIR = os.path.dirname(ROOT)
sys.path.insert(0, WORK_DIR)


class converter:
    def __init__(self, path_csv):
        self.path_csv = pd.read_csv(os.path.join(WORK_DIR, path_csv))
        self.folder_save = '/'.join(path_csv.split('/')[:2])
        self.scale = MinMaxScaler(feature_range = (0, 1))

    def __writeUserSampleFile(self):
        self.path_csv = self.path_csv.to_dict('list')
        df = pd.DataFrame(self.scale.fit_transform(np.array(self.path_csv['Close']).reshape(-1, 1)))
        if not os.path.exists(os.path.join(WORK_DIR, self.folder_save)):
            os.makedirs(os.path.join(WORK_DIR, self.folder_save))
        df.to_csv(os.path.join(WORK_DIR, self.folder_save, 'user_1_sample_1_money_A.csv'), index=False, header=False)

    def __writeAnnotationUserSampleFile(self):
        annotation = ['Close' for i in range(len(self.path_csv['Close']))]
        df = pd.DataFrame(annotation)
        df.to_csv(os.path.join(WORK_DIR, self.folder_save, 'Annotation_user_1_sample_1_money_A.csv'), index=False,
                  header=False)

    def __call__(self, *args, **kwargs):
        self.__writeUserSampleFile()
        self.__writeAnnotationUserSampleFile()


if __name__ == '__main__':
    custom_datasets = converter('datasets/money/BTC-USD.csv')
    custom_datasets()
