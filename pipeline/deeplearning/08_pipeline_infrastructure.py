import pandas as pd
import numpy as np
import time

def aggressive_downcasting(df):
    """
    Iterates through all numerical columns and tightly aggressively downcasts 
    precision types to prevent kaggle container memory failures.
    """
    int_cols = df.select_dtypes(include=['int64', 'int32']).columns
    float_cols_64 = df.select_dtypes(include=['float64']).columns
    float_cols_32 = df.select_dtypes(include=['float32']).columns
    
    for col in int_cols:
        col_min = df[col].min()
        col_max = df[col].max()
        if col_min > np.iinfo(np.int16).min and col_max < np.iinfo(np.int16).max:
            df[col] = df[col].astype(np.int16)
        elif col_min > np.iinfo(np.int32).min and col_max < np.iinfo(np.int32).max:
            df[col] = df[col].astype(np.int32)
            
    for col in float_cols_64:
        df[col] = df[col].astype(np.float32)

    return df

class MemoryMonitor:
    def __init__(self, step_name=""):
        self.step_name = step_name
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        print(f"[START] Executing: {self.step_name}")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        print(f"[END] Completed: {self.step_name} in {duration:.2f} seconds.")
