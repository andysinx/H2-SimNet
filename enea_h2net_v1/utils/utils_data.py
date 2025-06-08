import pandas as pd
import numpy as np
from itertools import cycle
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")
import os

def read_and_print_dataset_description(list_name_files_path: np.ndarray):
    datasets = []
    for name_file_path in list_name_files_path:
        dataset = pd.read_csv(name_file_path)
        datasets.append(dataset)
        print("Dataset " + name_file_path  + ":")
        print("\nDescription dataset:")
        dataset.Time = pd.to_datetime(dataset.Time, infer_datetime_format=True, unit='s')
        dataset['Time'] = dataset['Time'].apply(lambda x: x.replace(year=2024,month=5, day=15))
        print(dataset)
        print(dataset.info())
        print(dataset.iloc[:, 1:7].shape)
        print(dataset.iloc[:, 1:7].describe())
    return datasets
    

def plot_time_series_graphics(dataset: np.ndarray):
    
    layout = dict(xaxis=dict(title='Time'), yaxis=dict(title='Pressure'))
    empty_folder = True
    
    files = os.listdir("./time_series_plot")
    
    if files[0].lower().endswith(('.png')):
        empty_folder = False  # folder not empty
        print("Folder time_series_plot not empty")
    
    if (empty_folder):

        for i in range(len(dataset)):
            
            fig = go.Figure(layout=layout)

            fig.add_trace(go.Scatter(x=dataset[i].Time, y=dataset[i].PS_1, 
                                    mode='markers',
                                    marker=dict(color='red')))
            fig.add_trace(go.Scatter(x=dataset[i].Time, y=dataset[i].PS_2, 
                                    mode='markers',
                                    marker=dict(color='blue')))
            fig.add_trace(go.Scatter(x=dataset[i].Time, y=dataset[i].PS_3, 
                                    mode='markers',
                                    marker=dict(color='green')))
            fig.add_trace(go.Scatter(x=dataset[i].Time, y=dataset[i].PS_4, 
                                    mode='markers',
                                    marker=dict(color='yellow')))
            fig.add_trace(go.Scatter(x=dataset[i].Time, y=dataset[i].PS_5, 
                                    mode='markers',
                                    marker=dict(color='orange')))

            names = cycle(['PS_1', 'PS_2','PS_3','PS_4','PS_5'])
            fig = fig.for_each_trace(lambda t:  t.update(name = next(names)))
            fig.write_image("time_series_plot/time_series_" + str(i) + ".png")   