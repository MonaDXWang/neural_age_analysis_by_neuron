import json

from pathlib import Path
from util import open_json, tuple_key, get_key
from neuron_info import npair, nclass

def rename_nondauers(nondauer):
    
    nondauer_original = open_json(nondauer)
    nondauer_renamed = {}

    #Renamed the dictionary with the appropriate developmental stage
    for dataset, connections in nondauer_original.items():
        if dataset == 'Dataset1':
            nondauer_renamed['L1_1'] = nondauer_original[dataset]
        
        elif dataset == 'Dataset2':
            nondauer_renamed['L1_2'] = nondauer_original[dataset]
        
        elif dataset == 'Dataset3':
            nondauer_renamed['L1_3'] = nondauer_original[dataset]

        elif dataset == 'Dataset4':
            nondauer_renamed['L1_4'] = nondauer_original[dataset]
        
        elif dataset == 'Dataset5':
            nondauer_renamed['L2'] = nondauer_original[dataset]

        elif dataset == 'Dataset6':
            nondauer_renamed['L3'] = nondauer_original[dataset]
        
        elif dataset == 'Dataset7':
            nondauer_renamed['adult_TEM'] = nondauer_original[dataset]

        elif dataset == 'Dataset8':
            nondauer_renamed['adult_SEM'] = nondauer_original[dataset]
        
        else:
            print('There are unknown dataset names')
            continue
    return nondauer_renamed

def input_json_formatter(file,name):
    
    original = open_json(file)
    connections = {}
    connections[name] = {}
    
    for i, con in enumerate(original):

        key = str(tuple_key(original[i]['partners'][0], original[i]['partners'][1]))
        if key not in connections[name].keys():
            connections[name][key] = 1
        
        else:
            connections[name][key] += 1
    
    return connections

def append_to_nondauers(path, nondauer, input):
    
    connections = {**nondauer, **input}

    output_path = f'{path}/transformed-connections.json'

    with open(output_path, 'w') as f:
        json.dump(connections, f, indent= 2)
    
    return output_path

def get_input_output(path, data, connection_type = 'cell-to-cell'):
    
    assert connection_type in ('cell-to-cell', 'neuron_pair', 'neuron_class')

    #open the json file
    data = open_json(data)

    total_input = {}
    total_output = {}

    for dataset in data.keys():
        total_input[dataset] = {}
        total_output[dataset] = {}
        for connection in data[dataset]:

            connection_temp1 = connection.strip('"()')
            connection_temp2 = connection_temp1.replace("'", "")
            neuron_list = connection_temp2.split(', ')
            
            pre = neuron_list[0]
            post = neuron_list[1]

            if connection_type == 'cell-to-cell':
                key1 = pre
                key2 = post
            
            elif connection_type == 'neuron_pair':
                key1 = npair(pre)
                key2 = npair(post)

            else:
                key1 = nclass(pre)
                key2 = nclass(post)
            
            if key1 not in total_output[dataset].keys():
                total_output[dataset][key1] = data[dataset][connection]
        
            else:
                total_output[dataset][key1] += data[dataset][connection]

            if key2 not in total_input[dataset].keys():
                total_input[dataset][key2] = data[dataset][connection]
        
            else:
                total_input[dataset][key2] += data[dataset][connection]
                
    total_input_path = Path(f'{path}/total_input_by_neurons.json')
    total_output_path = Path(f'{path}/total_output_by_neurons.json')

    with open(total_input_path, 'w') as f:
        json.dump(total_input, f, indent= 2)
    
    with open(total_output_path, 'w') as f:
        json.dump(total_output, f, indent= 2)

    return total_input_path, total_output_path

def get_connections(path, data, connection_type = 'cell-to-cell'):
    
    assert connection_type in ('cell-to-cell', 'neuron_pair', 'neuron_class')

    #open the json file
    data = open_json(data)

    connections = {}

    connection_list = []

    for dataset in data.keys():
        connections[dataset] = {}
        for connection in data[dataset]:

            connection_temp1 = connection.strip('"()')
            connection_temp2 = connection_temp1.replace("'", "")
            neuron_list = connection_temp2.split(', ')
            
            pre = neuron_list[0]
            post = neuron_list[1]

            if connection_type == 'cell-to-cell':
                key = get_key(pre, post)
            
            elif connection_type == 'neuron_pair':
                key = get_key(npair(pre), npair(post))

            else:
                key = get_key(nclass(pre), nclass(post))

            if key not in connections[dataset].keys():
                connections[dataset][key] = data[dataset][connection]
        
            else:
                connections[dataset][key] += data[dataset][connection]

    for dataset, connection in connections.items():
        for key in connection:
            if key not in connection_list:
                connection_list.append(key)

    #Help check numbers with nemanode to make sure it's currectly added
    #print(connections['L1_1']['SAA-AVA'], connections['L1_2']['SAA-AVA'], connections['L1_3']['SAA-AVA'], connections['L2']['SAA$AVA'])

    output_path = f'{path}/connections.json'

    with open(output_path, 'w') as f:
        json.dump(connections, f, indent= 2)

    return output_path, connection_list