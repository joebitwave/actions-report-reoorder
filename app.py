import streamlit as st
import pandas as pd
import base64
from io import BytesIO
import time
import random
from itertools import permutations
from datetime import datetime

# Streamlit app title
st.title("Actions Report Reorder App")

# File uploader
uploaded_file = st.file_uploader("Upload your Actions Report CSV", type=["csv"])

if uploaded_file is not None:
    # Read the CSV file
    try:
        df = pd.read_csv(uploaded_file, encoding='utf-8')
        df = df.fillna('')  # Replace NaN with empty string
    except Exception as e:
        st.error(f"Error reading CSV: {str(e)}")
        st.write("Please ensure the CSV is properly formatted with headers.")
        st.stop()

    # Expected columns
    expected_columns = ['timestamp', 'action', 'asset', 'assetUnitAdj', 'assetBalance', 'inventory']
    
    # Check if expected columns are present
    missing_cols = [col for col in expected_columns if col not in df.columns]
    if missing_cols:
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
        
        # Log non-numeric rows
        non_numeric_rows = df[df['assetUnitAdj'].isna() | df['assetBalance'].isna()]
        if not non_numeric_rows.empty:
            st.warning(f"Found {len(non_numeric_rows)} rows with non-numeric assetUnitAdj or assetBalance. Set to 0.")
            st.write("Sample non-numeric rows:", non_numeric_rows[['timestamp', 'inventory', 'asset']].head().to_dict())

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
                return False

        for idx in indices:
            row = group.iloc[idx]
            action = row['action'].lower()
            try:
                units = float(row['assetUnitAdj'])
                current_balance = float(row['assetBalance'])
            except (ValueError, TypeError):
                return False

            # Use assetUnitAdj directly, accounting for its sign
            expected_balance = balance + units
            if abs(expected_balance - current_balance) > tolerance:
                return False
            balance = current_balance
        return True

    # Optimized greedy permutation finder
    def find_valid_permutation(group, prev_balance=None, prioritize_initial=False, max_rows=10):
        indices = list(range(len(group)))
        if len(indices) > max_rows:
            # Greedy approach with backtracking
            ordered = []
            remaining = indices.copy()
            balance = prev_balance
            if prioritize_initial:
                initial_indices = [i for i in indices if abs(float(group.iloc[i]['assetUnitAdj']) - float(group.iloc[i]['assetBalance'])) < 1e-6]
                if initial_indices:
                    start_idx = initial_indices[0]
                    ordered.append(start_idx)
                    remaining.remove(start_idx)
                    balance = float(group.iloc[start_idx]['assetBalance'])

            while remaining:
                next_idx = None
                for idx in remaining:
                    row = group.iloc[idx]
                    try:
                        units = float(row['assetUnitAdj'])
                        current_balance = float(row['assetBalance'])
                        expected_balance = balance + units
                        if abs(expected_balance - current_balance) < 1e-6:
                            next_idx = idx
                            break
                    except (ValueError, TypeError):
                        continue
                if next_idx is None:
                    break
                ordered.append(next_idx)
                remaining.remove(next_idx)
                balance = float(group.iloc[next_idx]['assetBalance'])

            ordered.extend(remaining)
            if is_valid_order(group, ordered, prev_balance):
                return [group.index[i] for i in ordered]
            return [group.index[i] for i in indices]  # Fallback to original order
        else:
            # Small groups: try permutations with prioritization
            if prioritize_initial:
                initial_indices = [i for i in indices if abs(float(group.iloc[i]['assetUnitAdj']) - float(group.iloc[i]['assetBalance'])) < 1e-6]
                for start_idx in initial_indices:
                    other_indices = [i for i in indices if i != start_idx]
                    for perm in permutations(other_indices):
                        test_perm = [start_idx] + list(perm)
                        if is_valid_order(group, test_perm, prev_balance):
                            return [group.index[i] for i in test_perm]
            for perm in permutations(indices):
                if is_valid_order(group, perm, prev_balance):
                    return [group.index[i] for i in perm]
            return [group.index[i] for i in indices]  # Fallback to original order

    # Reorder the DataFrame
    def reorder_dataframe(df, timeout=60):
        df = df.copy()
        # Convert timestamps to datetime
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            df = df.sort_values(by='timestamp', kind='mergesort')
        except Exception as e:
            st.error(f"Error sorting by timestamp: {str(e)}")
            st.stop()

        reordered_indices = []
        balance_tracker = {}
        start_time = time.time()
        
        grouped = df.groupby(['timestamp', 'inventory'])
        unique_timestamps = sorted(df['timestamp'].unique())
        progress_bar = st.progress(0)
        total_steps = len(unique_timestamps)
        
        for step, timestamp in enumerate(unique_timestamps):
            timestamp_rows = df[df['timestamp'] == timestamp]
            for inventory in sorted(timestamp_rows['inventory'].unique()):
                try:
                    t_inv_group = grouped.get_group((timestamp, inventory))
                except KeyError:
                    t_inv_group = timestamp_rows[timestamp_rows['inventory'] == inventory]
                    reordered_indices.extend(t_inv_group.index)
                    st.warning(f"Group not found for timestamp {timestamp}, inventory {inventory}. Using original order.")
                    continue

                t_inv_group = t_inv_group.sort_index()
                asset_groups = t_inv_group.groupby('asset')

                for asset in sorted(t_inv_group['asset'].unique()):
                    try:
                        t_group = asset_groups.get_group(asset)
                    except KeyError:
                        t_group = t_inv_group[t_inv_group['asset'] == asset]
                        reordered_indices.extend(t_group.index)
                        st.warning(f"Asset group not found for {inventory}, {asset} at {timestamp}. Using original order.")
                        continue

                    buy_sell_rows = t_group[t_group['action'].str.lower().isin(["buy", "sell"])]
                    non_buy_sell_rows = t_group[~t_group['action'].str.lower().isin(["buy", "sell"])]

                    prev_balance = balance_tracker.get((inventory, asset), None)
                    is_earliest = timestamp == df[(df['inventory'] == inventory) & (df['asset'] == asset)]['timestamp'].min()

                    if len(buy_sell_rows) > 1:
                        valid_order = find_valid_permutation(buy_sell_rows, prev_balance, prioritize_initial=is_earliest)
                        buy_sell_ordered = buy_sell_rows.loc[valid_order]
                        try:
                            balance_tracker[(inventory, asset)] = float(buy_sell_ordered.iloc[-1]['assetBalance'])
                        except (ValueError, TypeError):
                            pass
                    elif len(buy_sell_rows) == 1:
                        row = buy_sell_rows.iloc[0]
                        try:
                            units = float(row['assetUnitAdj'])
                            current_balance = float(row['assetBalance'])
                            if prev_balance is not None:
                                expected_balance = prev_balance + units
                                if abs(expected_balance - current_balance) > 1e-6:
                                    st.warning(
                                        f"Balance mismatch for {inventory}, {asset} at {timestamp}: "
                                        f"expected {expected_balance}, got {current_balance}. "
                                        f"Prev balance: {prev_balance}, Units: {units}, Action: {row['action']}"
                                    )
                            buy_sell_ordered = buy_sell_rows
                            balance_tracker[(inventory, asset)] = current_balance
                        except (ValueError, TypeError):
                            buy_sell_ordered = buy_sell_rows
                    else:
                        buy_sell_ordered = buy_sell_rows

                    if not non_buy_sell_rows.empty:
                        combined = pd.concat([buy_sell_ordered, non_buy_sell_rows]).sort_index()
                        reordered_indices.extend(combined.index)
                    else:
                        reordered_indices.extend(buy_sell_ordered.index)

                # Check for timeout
                if time.time() - start_time > timeout:
                    st.warning(f"Processing timeout at timestamp {timestamp}. Including remaining rows in original order.")
                    remaining_indices = df.index[~df.index.isin(reordered_indices)]
                    reordered_indices.extend(remaining_indices)
                    break

            # Update progress
            progress_bar.progress((step + 1) / total_steps)

        remaining_indices = df.index[~df.index.isin(reordered_indices)]
        if remaining_indices.size > 0:
            st.warning(f"Found {len(remaining_indices)} rows not processed. Including them in original order.")
            reordered_indices.extend(remaining_indices)

        if len(reordered_indices) != len(df):
            st.error(f"Row count mismatch: expected {len(df)} rows, got {len(reordered_indices)}.")
            st.stop()

        reordered_df = df.loc[reordered_indices].reset_index(drop=True)
        
        # Check monotonicity
        if not reordered_df['timestamp'].astype(str).is_monotonic_increasing:
            st.warning("Output rows are not in strict chronological order.")

        # Log processing time
        st.write(f"Processing time: {time.time() - start_time:.2f} seconds")
        
        return reordered_df

    # Process the CSV
    try:
        reordered_df = reorder_dataframe(df)
        st.write(f"Reordered CSV Data ({len(reordered_df)} rows):")
        st.dataframe(reordered_df)

        def get_csv_download_link(df, filename="reordered_actions_report.csv"):
            csv_buffer = BytesIO()
            df.to_csv(csv_buffer, index=False)
            csv_buffer.seek(0)
            b64 = base64.b64encode(csv_buffer.read()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download Reordered CSV</a>'
            return href

        st.markdown(get_csv_download_link(reordered_df), unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error processing the CSV: {str(e)}")
        st.write("Available columns:", list(df.columns))
        st.write("Please ensure the CSV has the required columns.")
