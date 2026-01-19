import pandas as pd
import numpy as np

class DataLoader:
    """
    Handles loading and merging of Tweet data and VIX data.
    """
    def __init__(self, tweets_path, vix_path):
        self.tweets_path = tweets_path
        self.vix_path = vix_path

    def load_and_clean_vix(self):
        print(f"Loading VIX from: {self.vix_path}")
        df_vix = pd.read_csv(self.vix_path)
        
        # --- FIX: Standardize Column Names ---
        # 1. Strip whitespace
        df_vix.columns = df_vix.columns.str.strip()
        
        # 2. Rename UPPERCASE to Title Case (The fix for your file)
        # We look for 'DATE' and make it 'Date', 'CLOSE' becomes 'Close'
        rename_map = {
            'DATE': 'Date', 
            'CLOSE': 'Close',
            'date': 'Date',
            'close': 'Close'
        }
        df_vix.rename(columns=rename_map, inplace=True)
        # -------------------------------------

        # Verify we have what we need
        if 'Date' not in df_vix.columns or 'Close' not in df_vix.columns:
             raise KeyError(f"Missing required columns. Found: {df_vix.columns.tolist()}")

        df_vix['Date'] = pd.to_datetime(df_vix['Date'])
        
        # Target: Next day's Close price (Shift -1)
        df_vix['target_value'] = df_vix['Close'].shift(-1)
        
        return df_vix.dropna()[['Date', 'Close', 'target_value']]

    def merge_data(self, df_tweets):
        df_vix = self.load_and_clean_vix()
        
        # Convert tweet dates to standard datetime
        df_tweets['date'] = pd.to_datetime(pd.to_datetime(df_tweets['date']).dt.date)
        
        # Merge
        print("Merging datasets...")
        merged = pd.merge(df_tweets, df_vix, left_on='date', right_on='Date', how='inner')
        print(f"Merge Complete. Shape: {merged.shape}")
        return merged