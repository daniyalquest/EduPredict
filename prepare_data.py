import pandas as pd
import json

def generate_mongo_json():
    # 1. Load the raw CSV
    try:
        df = pd.read_csv('Student_Performance.csv')
    except FileNotFoundError:
        print("Error: Student_Performance.csv not found.")
        return

    # 2. Cleaning & Formatting
    # Standardize column names (lowercase and underscore)
    df.columns = [c.strip().replace(' ', '_').lower() for c in df.columns]

    # Fill numeric missing values with median
    numeric_cols = df.select_dtypes(include=['number']).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    # Fill categorical missing values with 'Unknown'
    cat_cols = df.select_dtypes(include=['object']).columns
    df[cat_cols] = df[cat_cols].fillna('Unknown')

    # 3. Save as JSON (MongoDB friendly)
    # Records format creates a list of dictionaries: [{}, {}, {}]
    output_file = 'cleaned.json'
    df.to_json(output_file, orient='records', indent=4)
    
    print(f"✓ Success! '{output_file}' created with {len(df)} records.")

if __name__ == "__main__":
    generate_mongo_json()