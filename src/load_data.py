import pandas as pd

def load_nab_data(path):
    df = pd.read_csv(path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    
    # NAB format: value column
    data = df['value'].values
    
    return data, df