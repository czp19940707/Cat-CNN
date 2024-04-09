import pandas as pd
import subprocess

if __name__ == '__main__':
    commands = []
    frame = pd.read_csv(r'T1_FGD.csv')
    frame = frame[~pd.isna(frame['Data path.pet.fgd'])]
    for index_ in frame.index:
        subject_information = frame.loc[index_, :]
        path = subject_information['Data path']
        command = 'python Crop.py -i {}'.format(path)
        commands.append(command)

    process = subprocess.Popen(['parallel', '-j', '32', '--gnu', ':::'] + commands)
    process.wait()
