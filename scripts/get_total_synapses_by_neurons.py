import data_organizer as do
import numpy as np
import pandas as pd

from pathlib import Path
from util import get_job_dir

job_dir = get_job_dir()
#change if needed 
target_neurons = ['RIA', 'ALA', 'AVA', 'AVB', 'AVE', 'AVD', 'RME', 'RIH', 'RIS', 'PVR']
connection_type = 'neuron_class' #cell-to-cell, neuron_pair
synapse_type = 'count' #count, size
name = 'twk-40(gf)'
input_data_path = Path(f'./input/{name}.json')
nondauers = Path(f'./input/nondauer_synapse_{synapse_type}.json') #nondauer_synapse_size.json, nondauer_synapse_count.json

nondauer_renamed = do.rename_nondauers(nondauers)
input_formatted = do.input_json_formatter(input_data_path, f'{name}')

merged_data = do.append_to_nondauers(job_dir, nondauer_renamed, input_formatted)

total_input, total_output = do.get_input_output(job_dir,merged_data, connection_type = connection_type)

df_all_input = pd.read_json(total_input)
df_all_output = pd.read_json(total_output)
df_all_input.index.name = 'Neuron'
df_all_output.index.name = 'Neuron'

df_input = df_all_input[df_all_input.index.isin(target_neurons)]
df_output = df_all_output[df_all_output.index.isin(target_neurons)]

with pd.ExcelWriter(f'{job_dir}/total_input_output_by_neurons.xlsx') as writer:
            df_input.to_excel (writer, sheet_name = 'total input', index = True)
            df_output.to_excel (writer, sheet_name = 'total output', index = True)
    