import re
import pandas as pd
import numpy as np

class TweetCleaner:
    """
    Handles cleaning and adds robust 'Topic Detection' for 5 distinct categories.
    """
    def __init__(self):
        self.url_pattern = re.compile(r'https?://\S+|www\.\S+')
        self.mention_pattern = re.compile(r'@\w+')
        self.special_chars = re.compile(r'[^a-zA-Z\s]')
        
        # --- NEW: Expanded Topic Keywords ---
        self.topics = {
            'economy': [
                'market', 'stock', 'tax', 'tariff', 'trade', 'job', 'economy', 'dollar', 
                'fed', 'reserve', 'powell', 'rate', 'gdp', 'inflation', 'china'
            ],
            'geopolitics': [
                'war', 'nuclear', 'missile', 'iran', 'korea', 'military', 'attack', 
                'russia', 'putin', 'xi', 'kim', 'taiwan', 'venezuela', 'world', 'treaty'
            ],
            'domestic': [
                'border', 'wall', 'ice', 'immigration', 'mexico', 'democrat', 'republican',
                'pelosi', 'biden', 'hillary', 'election', 'vote', 'fraud', 'court', 
                'supreme', 'shutdown', 'impeachment', 'witch', 'hunt', 'police', 'law', 'order'
            ],
            'corporate': [
                'amazon', 'boeing', 'gm', 'ford', 'toyota', 'apple', 'google', 'facebook', 
                'twitter', 'musk', 'company', 'companies', 'media', 'cnn', 'fake', 'news'
            ]
        }

    def clean_text(self, text):
        if not isinstance(text, str): return ""
        text = text.lower()
        text = self.url_pattern.sub('', text)
        text = self.mention_pattern.sub('', text)
        text = self.special_chars.sub('', text)
        return ' '.join(text.split())

    def detect_topic(self, text, topic_name):
        """Returns 1 if any keyword matches, 0 otherwise"""
        keywords = self.topics[topic_name]
        return 1 if any(word in text for word in keywords) else 0

    def preprocess_dataframe(self, df, text_column):
        print(f"Cleaning {len(df)} tweets and tagging expanded topics...")
        
        # 1. Clean Text
        df['cleaned_text'] = df[text_column].apply(self.clean_text)
        
        # 2. Feature Engineering: Length
        df['tweet_length'] = df['cleaned_text'].apply(len)
        
        # 3. Feature Engineering: Specific Topics
        df['is_economy'] = df['cleaned_text'].apply(lambda x: self.detect_topic(x, 'economy'))
        df['is_geo'] = df['cleaned_text'].apply(lambda x: self.detect_topic(x, 'geopolitics'))
        df['is_domestic'] = df['cleaned_text'].apply(lambda x: self.detect_topic(x, 'domestic'))
        df['is_corporate'] = df['cleaned_text'].apply(lambda x: self.detect_topic(x, 'corporate'))
        
        # 4. Feature Engineering: "Other"
        # If it's not any of the above, it's "Other"
        df['is_other'] = ((df['is_economy'] + df['is_geo'] + df['is_domestic'] + df['is_corporate']) == 0).astype(int)
        
        # Filter empty
        df = df[df['cleaned_text'] != '']
        
        return df