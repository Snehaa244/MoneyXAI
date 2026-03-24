import json

def extract_columns_from_notebook(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    for cell in nb.get('cells', []):
        if cell.get('cell_type') == 'code':
            source = "".join(cell.get('source', []))
            if "df.columns" in source or "df.head()" in source or "raw_df = pd.read_csv" in source:
                for output in cell.get('outputs', []):
                    if 'data' in output and 'text/plain' in output['data']:
                        print("Output from cell:")
                        print("".join(output['data']['text/plain']))
                    if 'text' in output:
                        print("Output text:")
                        print("".join(output['text']))

extract_columns_from_notebook('C:/Users/HP/Downloads/MoneyXAI-main/MoneyXAI-main/MoneyXAI.ipynb')
