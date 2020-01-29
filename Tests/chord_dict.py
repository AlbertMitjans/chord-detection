import os
import pandas as pd

chord_dict = pd.read_excel(os.path.join(os.getcwd(), 'dataset', 'guitar_chords.xlsx'))

for idx, val in enumerate(chord_dict['Fret']):
    if val == 'x':
        chord_dict['Fret'][idx] = -1
        chord_dict['Finger'][idx] = -1
        chord_dict['Note'][idx] = -1

chord_dict.to_csv(path_or_buf=os.path.join(os.getcwd(), 'dataset', 'guitar_chords.csv'), index=False)

