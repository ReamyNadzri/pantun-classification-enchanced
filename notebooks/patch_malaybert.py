import json

def patch_notebook():
    notebook_path = "c:/Users/rahim/pantun-classification-enchanced-2/notebooks/train_malaybert.ipynb"
    
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # Patch the MODEL_NAME cell to use the correct public model
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            new_source = []
            changed = False
            for line in cell['source']:
                if 'malay-huggingface/bert-base-bahasa-cased' in line:
                    line = line.replace(
                        'malay-huggingface/bert-base-bahasa-cased',
                        'mesolitica/bert-base-standard-bahasa-cased'
                    )
                    changed = True
                new_source.append(line)
            if changed:
                cell['source'] = new_source
                print("Fixed MODEL_NAME in cell.")

    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=4)
    print("Done!")

if __name__ == "__main__":
    patch_notebook()
