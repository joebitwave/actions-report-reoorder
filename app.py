import streamlit as st
import pandas as pd
import base64
from io import BytesIO
from itertools import permutations
import random

# Streamlit app title
st.title("Actions Report Reorder App")

# File uploader
uploaded_file = st.file_uploader("Upload your Actions Report CSV", type=["csv"])

if uploaded_file is not None:
    # Read the CSV file
    try:
        df = pd.read_csv(uploaded_file, encoding='utf-8')
        # Ensure string types for key columns and handle NaN
        df.iloc[:, 3] = df.iloc[:, 3].astype(str).fillna('')
        df.iloc[:, 9] = df.iloc[:, 9].astype(str).fillna('Unknown')
        df.iloc[:, 27] = df.iloc[:, 27].astype(str).fillna('Unknown')
        df.iloc[:, 4] = df.iloc[:, 4].astype(str).fillna('')
    except Exception as e:
        st.error(f"Error reading CSV: {str(e)}")
        st.write("Please ensure the CSV is properly formatted with headers.")
        st.stop()
    
    # Debug: Display unique timestamp and inventory values
    st.write("Unique timestamp values:", sorted(df.iloc[:, 3].unique().tolist()))
    st.write("Unique inventory values:", sorted(df.iloc[:, 27].unique().tolist()))
    
    # Check for missing or empty inventory
    invalid_inventory_rows = df[df.iloc[:, 27].str.strip() == '']
    if not invalid_inventory_rows.empty:
        st.warning(f"Found {len(invalid_inventory_rows)} rows with missing or empty inventory values. Assigned 'Unknown'.")
        st.write("Sample invalid inventory rows:", invalid_inventory_rows.iloc[:, [3, 27, 9]].head().to_dict())
        df.loc[df.iloc[:, 27].str.strip() == '', df.columns[27]] = 'Unknown'
    
    # Display original data
    st.write(f"Original CSV Data ({len(df)} rows):")
    st.dataframe(df)
    
    # Validate required columns
    required_cols = [3, 4, 9, 11, 12, 27]  # timestamp, action, asset, assetUnitAdj, assetBalance, inventory
    if len(df.columns) < max(required_cols) + 1:
        st.error(f"CSV has too few columns. Expected at least {max(required_cols) + 1}, found {len(df.columns)}.")
        st.write("Available columns:", list(df.columns))
        st.stop()
    
    # Check if 'action' column exists
    if df.columns[4] != 'action':
        st.error("CSV is missing the 'action' column (expected in Column E).")
        st.write("Available columns:", list(df.columns))
        st.stop()
    
    # Function to validate balance transitions
    def is_valid_order(group, indices, prev_balance=None):
        balance = prev_balance if prev_balance is not None else None
        if balance is None:
            try:
                balance = float(group.iloc[indices[0]].iloc[12])  # assetBalance
            except (ValueError, TypeError):
                return False
        
        for idx in indices:
            row = group.iloc[idx]
            action
