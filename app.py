import streamlit as st
import pandas as pd
import base64
from io import BytesIO
import time
import random
from itertools import permutations
from datetime import datetime
import logging
import sys

# Set up logging with fallback to stdout if file is not writable
try:
    logging.basicConfig(
        filename='streamlit_app.log',
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
except Exception as e:
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logging.error(f"Failed to set up file logging: {str(e)}")

# Streamlit app title
st.title("Actions Report Reorder App")

# File uploader
uploaded_file = st.file_uploader("Upload your Actions Report CSV", type=["csv"])

if uploaded_file is not None:
    # Display processing message
    status_text = st.empty()
    status_text.write("Processing uploaded CSV...")

    # Read the CSV file
    try:
        logging.info("Reading CSV file")
        df = pd.read_csv(uploaded_file, encoding='utf-8')
        df = df.fillna('')  # Replace NaN with empty string
        logging.info(f"CSV read successfully: {len(df)} rows")
    except Exception as e:
        logging.error(f"Error reading CSV: {str(e)}")
        status_text.empty()
        st.error(f"Error reading CSV: {str(e)}")
        st.write("Please ensure the CSV is properly formatted with headers.")
        st.stop()

    # Expected columns
    expected_columns = ['timestamp', 'action', 'asset', 'assetUnitAdj', 'assetBalance', 'inventory']
    
    # Check if expected columns are present
    missing_cols = [col for col in expected_columns if col not in df.columns]
    if missing_cols:
        status_text.empty()
        st.warning(f"Missing columns: {missing_cols}. Please map the columns below.")
        col_mapping = {}
        for col in expected_columns:
            if col in missing_cols:
                col_mapping[col] = st.selectbox(f"Select column for {col}", options=[''] + list(df.columns), key=col)
            else:
                col_mapping[col] = col
        if any(v == '' for v in col_mapping.values()):
            st.error("Please map all required columns.")
            st.stop()
        df = df.rename(columns={v: k for k, v in col_mapping.items() if v != k})
        logging.info("Columns mapped successfully")

    # Ensure string types and convert numeric columns
    df['timestamp'] = df['timestamp'].astype(str)
    df['action'] = df['action'].astype(str)
    df['asset'] = df['asset'].astype(str).fillna('Unknown')
    df['inventory'] = df['inventory'].astype(str).fillna('Unknown')
    
    # Convert numeric columns
    df['assetUnitAdj'] = pd.to_numeric(df['assetUnitAdj'], errors='coerce').fillna(0)
    df['assetBalance'] = pd.to_numeric(df['assetBalance'], errors='coerce').fillna(0)
    
    # Debug information in an expander
    with st.expander("Debug Information"):
        st.write("Column names:", list(df.columns))
        st.write("Unique timestamp values:", sorted(df['timestamp'].unique().tolist()))
        st.write("Unique inventory values:", sorted(df['inventory'].unique().tolist()))
        st.write(f"Total rows: {len(df)}")
        
        # Log non-numeric rows
        non_numeric_rows = df[df['assetUnitAdj'].isna() | df['assetBalance'].isna()]
        if not non_numeric_rows.empty:
            st.warning(f"Found {len(non_numeric_rows)} rows with non-numeric assetUnitAdj or assetBalance. Set to 0.")
            st.write("Sample non-numeric rows:", non_numeric_rows[['timestamp', 'inventory', 'asset']].head().to_dict())
            logging.warning(f"Non-numeric rows found: {len(non_numeric_rows)}")

    # Display original data
    st.write(f"Original CSV Data ({len(df)} rows):")
    st.dataframe(df)

    # Function to validate balance transitions
    def is_valid_order(group, indices, prev_balance=None, tolerance=1e-6):
        balance = prev_balance if prev_balance is not None else None
        if balance is None:
            try:
                balance = float(group.iloc[indices[0]]['assetBalance'])
            except (ValueError, TypeError):
                logging.error(f"Invalid initial balance for group: {group.iloc[indices[0]]}")
                return False

        for idx in indices:
            row = group.iloc[idx]
            try:
                units = float(row['assetUnitAdj'])
                current_balance = float(row['assetBalance'])
            except (ValueError, TypeError):
                logging.error(f"Invalid units or balance for row: {row}")
                return False

            expected_balance = balance + units
            if abs(expected_balance - current_balance) > tolerance:
                logging.warning(
                    f"Balance validation failed: expected {expected_balance}, got {current_balance}, "
                    f"units {units}, action {row['action']}"
                )
                return False
            balance = current_balance
        return True

    # Optimized greedy permutation finder
    def find_valid_permutation(group, prev_balance=None, prioritize_initial=False, max_rows=3):
        indices = list(range(len(group)))
        if len(indices) > max_rows:
            logging.info(f"Large group ({len(indices)} rows), using greedy approach")
            ordered = []
            remaining = indices.copy()
            balance = prev_balance
            if prioritize_initial:
                initial_indices = [
                    i for i in indices 
                    if abs(float(group.iloc[i]['assetUnitAdj']) - float(group.iloc[i]['assetBalance'])) < 1e-6
                ]
                if initial_indices:
                    start_idx = initial_indices[0]
                    ordered
