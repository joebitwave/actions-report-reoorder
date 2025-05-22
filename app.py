import streamlit as st
import pandas as pd
import base64
from io import BytesIO
import time
import random
import logging
import sys

# Set up logging with fallback to stdout
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
        # Use chunksize for large CSVs
        df_chunks = pd.read_csv(uploaded_file, encoding='utf-8', chunksize=10000)
        df = pd.concat([chunk.fillna('') for chunk in df_chunks])
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
    try:
        logging.info("Converting column types")
        df['timestamp'] = df['timestamp'].astype(str)
        df['action'] = df['action'].astype(str)
        df['asset'] = df['asset'].astype(str).fillna('Unknown')
        df['inventory'] = df['inventory'].astype(str).fillna('Unknown')
        df['assetUnitAdj'] = pd.to_numeric(df['assetUnitAdj'], errors='coerce').fillna(0)
        df['assetBalance'] = pd.to_numeric(df['assetBalance'], errors='coerce').fillna(0)
        logging.info("Column types converted")
    except Exception as e:
        logging.error(f"Error converting column types: {str(e)}")
        status_text.empty()
        st.error(f"Error converting column types: {str(e)}")
        st.stop()

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

    # Optimized greedy permutation finder (no permutations, only greedy)
    def find_valid_permutation(group, prev_balance=None, prioritize_initial=False, max_rows=0):
        indices = list(range(len(group)))
        logging.info(f"Processing group with {len(indices)} rows, using greedy approach")
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
                    expected_balance = balance + units if balance is not None else current_balance
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
        logging.warning(f"No valid order found for group, returning original order")
        return [group.index[i] for i in indices]

    # Reorder the DataFrame
    def reorder_dataframe(df, timeout=60):
        df = df.copy()
        logging.info("Starting DataFrame reordering")
        # Convert timestamps to datetime
        try:
            logging.info("Converting timestamps")
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            invalid_timestamps = df[df['timestamp'].isna()]
            if not invalid_timestamps.empty:
                logging.warning(f"Found {len(invalid_timestamps)} rows with invalid timestamps")
            df = df.sort_values(by='timestamp', kind='mergesort')
            logging.info("Timestamps converted and sorted")
        except Exception as e:
            logging.error(f"Error sorting by timestamp: {str(e)}")
            st.error(f"Error sorting by timestamp: {str(e)}")
            st.stop()

        reordered_indices = []
        balance_tracker = {}
        start_time = time.time()
        
        try:
            grouped = df.groupby(['timestamp', 'inventory'])
            unique_timestamps = sorted(df['timestamp'].dropna().unique())
            progress_bar = st.progress(0)
            total_steps = len(unique_timestamps)
            
            for step, timestamp in enumerate(unique_timestamps):
                logging.info(f"Processing timestamp: {timestamp}")
                timestamp_rows = df[df['timestamp'] == timestamp]
                for inventory in sorted(timestamp_rows['inventory'].unique()):
                    logging.info(f"Processing inventory: {inventory}")
                    try:
                        t_inv_group = grouped.get_group((timestamp, inventory))
                        logging.info(f"Inventory group size: {len(t_inv_group)} rows")
                    except KeyError:
                        t_inv_group = timestamp_rows[timestamp_rows['inventory'] == inventory]
                        reordered_indices.extend(t_inv_group.index)
                        logging.warning(f"Group not found for timestamp {timestamp}, inventory {inventory}")
                        st.warning(f"Group not found for timestamp {timestamp}, inventory {inventory}. Using original order.")
                        continue

                    t_inv_group = t_inv_group.sort_index()
                    asset_groups = t_inv_group.groupby('asset')

                    for asset in sorted(t_inv_group['asset'].unique()):
                        logging.info(f"Processing asset: {asset}")
                        try:
                            t_group = asset_groups.get_group(asset)
                            logging.info(f"Asset group size: {len(t_group)} rows")
                        except KeyError:
                            t_group = t_inv_group[t_inv_group['asset'] == asset]
                            reordered_indices.extend(t_group.index)
                            logging.warning(f"Asset group not found for {inventory}, {asset} at {timestamp}")
                            st.warning(f"Asset group not found for {inventory}, {asset} at {timestamp}. Using original order.")
                            continue

                        buy_sell_rows = t_group[t_group['action'].str.lower().isin(["buy", "sell"])]
                        non_buy_sell_rows = t_group[~t_group['action'].str.lower().isin(["buy", "sell"])]

                        prev_balance = balance_tracker.get((inventory, asset), None)
                        is_earliest = timestamp == df[(df['inventory'] == inventory) & (df['asset'] == asset)]['timestamp'].min()

                        if len(buy_sell_rows) > 1:
                            logging.info(f"Ordering {len(buy_sell_rows)} buy/sell rows")
                            valid_order = find_valid_permutation(buy_sell_rows, prev_balance, prioritize_initial=is_earliest)
                            buy_sell_ordered = buy_sell_rows.loc[valid_order]
                            try:
                                balance_tracker[(inventory, asset)] = float(buy_sell_ordered.iloc[-1]['assetBalance'])
                            except (ValueError, TypeError):
                                logging.error(f"Failed to update balance for {inventory}, {asset}")
                        elif len(buy_sell_rows) == 1:
                            row = buy_sell_rows.iloc[0]
                            try:
                                units = float(row['assetUnitAdj'])
                                current_balance = float(row['assetBalance'])
                                if prev_balance is not None:
                                    expected_balance = prev_balance + units
                                    if abs(expected_balance - current_balance) > 1e-6:
                                        logging.warning(
                                            f"Balance mismatch for {inventory}, {asset} at {timestamp}: "
                                            f"expected {expected_balance}, got {current_balance}. "
                                            f"Prev balance: {prev_balance}, Units: {units}, Action: {row['action']}"
                                        )
                                        st.warning(
                                            f"Balance mismatch for {inventory}, {asset} at {timestamp}: "
                                            f"expected {expected_balance}, got {current_balance}"
                                        )
                                buy_sell_ordered = buy_sell_rows
                                balance_tracker[(inventory, asset)] = current_balance
                            except (ValueError, TypeError):
                                buy_sell_ordered = buy_sell_rows
                                logging.error(f"Invalid data in single row for {inventory}, {asset}")
                        else:
                            buy_sell_ordered = buy_sell_rows

                        if not non_buy_sell_rows.empty:
                            combined = pd.concat([buy_sell_ordered, non_buy_sell_rows]).sort_index()
                            reordered_indices.extend(combined.index)
                        else:
                            reordered_indices.extend(buy_sell_ordered.index)

                    # Check for timeout
                    elapsed_time = time.time() - start_time
                    if elapsed_time > timeout:
                        logging.warning(f"Timeout at timestamp {timestamp} after {elapsed_time:.2f} seconds")
                        st.warning(f"Processing timeout at timestamp {timestamp}. Including remaining rows in original order.")
                        remaining_indices = df.index[~df.index.isin(reordered_indices)]
                        reordered_indices.extend(remaining_indices)
                        break

                if elapsed_time > timeout:
                    break

                # Update progress
                progress_bar.progress((step + 1) / total_steps)

        except Exception as e:
            logging.error(f"Error during group processing: {str(e)}")
            st.warning(f"Error during processing: {str(e)}. Including remaining rows in original order.")
            remaining_indices = df.index[~df.index.isin(reordered_indices)]
            reordered_indices.extend(remaining_indices)

        remaining_indices = df.index[~df.index.isin(reordered_indices)]
        if remaining_indices.size > 0:
            logging.warning(f"Unprocessed rows: {len(remaining_indices)}")
            st.warning(f"Found {len(remaining_indices)} rows not processed. Including them in original order.")
            reordered_indices.extend(remaining_indices)

        if len(reordered_indices) != len(df):
            logging.error(f"Row count mismatch: expected {len(df)}, got {len(reordered_indices)}")
            st.error(f"Row count mismatch: expected {len(df)} rows, got {len(reordered_indices)}.")
            st.stop()

        reordered_df = df.loc[reordered_indices].reset_index(drop=True)
        
        # Check monotonicity
        if not reordered_df['timestamp'].astype(str).is_monotonic_increasing:
            logging.warning("Output rows not in chronological order")
            st.warning("Output rows are not in strict chronological order.")

        # Log processing time
        processing_time = time.time() - start_time
        logging.info(f"Processing completed in {processing_time:.2f} seconds")
        status_text.empty()
        st.write(f"Processing time: {processing_time:.2f} seconds")
        
        return reordered_df

    # Process the CSV
    try:
        logging.info("Starting CSV processing")
        reordered_df = reorder_dataframe(df, timeout=60)
        logging.info("CSV processing completed")
        status_text.empty()
        st.write(f"Reordered CSV Data ({len(reordered_df)} rows):")
        st.dataframe(reordered_df)

        def get_csv_download_link(df, filename="reordered_actions_report.csv"):
            try:
                csv_buffer = BytesIO()
                df.to_csv(csv_buffer, index=False)
                csv_buffer.seek(0)
                b64 = base64.b64encode(csv_buffer.read()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download Reordered CSV</a>'
                return href
            except Exception as e:
                logging.error(f"Error generating download link: {str(e)}")
                st.error(f"Error generating download link: {str(e)}")
                return None

        download_link = get_csv_download_link(reordered_df)
        if download_link:
            st.markdown(download_link, unsafe_allow_html=True)
            logging.info("Download link generated")
        else:
            st.error("Failed to generate download link.")
    except Exception as e:
        logging.error(f"Error processing CSV: {str(e)}")
        status_text.empty()
        st.error(f"Error processing the CSV: {str(e)}")
        st.write("Available columns:", list(df.columns))
        st.write("Please ensure the CSV has the required columns.")
        # Fallback: provide original CSV as output
        st.write("Providing original CSV as fallback output:")
        download_link = get_csv_download_link(df, filename="original_actions_report.csv")
        if download_link:
            st.markdown(download_link, unsafe_allow_html=True)
