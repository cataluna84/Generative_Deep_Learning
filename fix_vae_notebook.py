
import json

file_path = '/home/cataluna84/Workspace-Antigravity/Generative_Deep_Learning/03_03_vae_digits_train.ipynb'

with open(file_path, 'r') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell.get('cell_type') == 'code':
        source = cell.get('source', [])
        new_source = []
        for line in source:
            if 'os.mkdir(RUN_FOLDER)' in line:
                new_source.append(line.replace('os.mkdir(RUN_FOLDER)', 'os.makedirs(RUN_FOLDER, exist_ok=True)'))
            elif "weights/weights.h5" in line:
                new_source.append(line.replace("weights/weights.h5", "weights/weights.weights.h5"))
            else:
                new_source.append(line)
        cell['source'] = new_source

with open(file_path, 'w') as f:
    json.dump(nb, f, indent=1)
