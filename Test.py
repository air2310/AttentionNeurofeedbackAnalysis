import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
import mne

# sample_data_folder = mne.datasets.sample.data_path()
# sample_data_raw_file = os.path.join(sample_data_folder, 'MEG', 'sample',
#                                     'sample_audvis_raw.fif')
# raw = mne.io.read_raw_fif(sample_data_raw_file, preload=True, verbose=False)

ten_twenty_montage = mne.channels.make_standard_montage('standard_1020')
print(ten_twenty_montage)

ten_twenty_montage.dig[ten_twenty_montage.ch_names.index('O2')+2]
ten_twenty_montage.dig[ten_twenty_montage.ch_names.index('PO3')+2]
ten_twenty_montage.dig[ten_twenty_montage.ch_names.index('PO4')+2]
ten_twenty_montage.dig[ten_twenty_montage.ch_names.index('PO7')+2]
ten_twenty_montage.dig[ten_twenty_montage.ch_names.index('PO8')+2]