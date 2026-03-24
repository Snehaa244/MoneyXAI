import json

def extract_notebook_info(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    code_cells = []
    for cell in nb.get('cells', []):
        if cell.get('cell_type') == 'code':
            source = "".join(cell.get('source', []))
            code_cells.append(source)
            
    with open('C:/Users/HP/Downloads/MoneyXAI-main/MoneyXAI-main/notebook_code.py', 'w', encoding='utf-8') as f:
        f.write("\n\n# --- CELL ---\n\n".join(code_cells))
        
extract_notebook_info('C:/Users/HP/Downloads/MoneyXAI-main/MoneyXAI-main/MoneyXAI.ipynb')
