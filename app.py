import streamlit as st
import pandas as pd
import base64
from io import BytesIO
from itertools import permutations

# Streamlit app title
st.title("Actions Report Reorder App")

# File uploader
uploaded_file = st.file_uploader("Upload your Actions Report CSV", type=["csv"])

if uploaded_file is not None:
    # Read the CSV file
    try:
        # Read CSV with utf-8 encoding and string dtypes for key columns
        df = pd.read_csv(uploaded_file, encoding='utf-8', dtype={3: str, 27: str, 9: str})
    except Exception as e:
        st.error(f"Error reading CSV: {str(e)}")
        st.write("Please ensure the CSV is properly formatted with headers.")
        st.stop()
    
    # Replace NaN with empty string to preserve valid values
    df = df.fillna('')
    
    # Debug: Display unique timestamp and inventory values
    st.write("Unique timestamp values:", sorted(df[df.columns[3]].unique().tolist()))
    st.write("Unique inventory values:", sorted(df[df.columns[27]].unique().tolist()))
    
    # Check for rows with missing or empty inventory
    invalid_inventory_rows = df[df[df.columns[27]].str.strip() == '']
    if not invalid_inventory_rows.empty:
        st.warning(f"Found {len(invalid_inventory_rows)} rows with missing or empty inventory values. Assigned 'Unknown'.")
        st.write("Sample invalid inventory rows:", invalid_inventory_rows[['timestamp', 'inventory', 'asset']].head().to_dict())
        df.loc[df[df.columns[27]].str.strip() == '', df.columns[27]] = 'Unknown'
    
    # Display original data
    st.write(f"Original CSV Data ({len(df)} rows):")
    st.dataframe(df)
    
    # Validate required columns
    required_cols = [3, 4, 9, 11, 12, 27]  # Indices for timestamp, action, asset, assetUnitAdj, assetBalance, inventory
    if len(df.columns) < max(required_cols) + 1:
        st.error(f"CSV has too few columns. Expected at least {max(required_cols) + 1}, found {len(df.columns)}.")
        st.write("Available columns:", list(df.columns))
        st.stop()
    
    # Check if 'action' column exists
    if "action" not in df.columns:
        st.error("CSV is missing the 'action' column (expected in Column E).")
        st.write("Available columns:", list(df.columns))
        st.stop()
    
    # Function to validate balance transitions for a sequence of buy/sell transactions
    def is_valid_order(group, indices, prev_balance=None):
        balance = prev_balance if prev_balance is not None else None
        if balance is None:
            try:
                balance = float(group.iloc[indices[0]][df.columns[12]])  # assetBalance (M)
            except (ValueError, TypeError):
                return False
        
        for i, idx in enumerate(indices):
            row = group.iloc[idx]
            action = row["action"].lower()  # action (E)
            try:
                units = float(row[df.columns[11]])  # assetUnitAdj (L)
                current_balance = float(row[df.columns[12]])  # assetBalance (M)
            except (ValueError, TypeError):
                return False  # Skip rows with non-numeric values
            
            if action == "buy":
                expected_balance = balance + units
            elif action == "sell":
                expected_balance = balance - units
            else:
                return False  # Should not happen as only buy/sell rows are passed
            
            if abs(expected_balance - current_balance) > 1e-6:  # Allow for float precision
                return False
            balance = current_balance
        return True

    # Function to find a valid permutation for tied-timestamp buy/sell rows
    def find_valid_permutation(group, prev_balance=None, prioritize_initial=False):
        indices = list(range(len(group)))
        if prioritize_initial:
            # Try permutations starting with rows where assetUnitAdj == assetBalance
            initial_indices = [i for i in indices if abs(float(group.iloc[i][df.columns[11]]) - float(group.iloc[i][df.columns[12]])) < 1e-6]
            for start_idx in initial_indices:
                other_indices = [i for i in indices if i != start_idx]
                for perm in permutations(other_indices):
                    test_perm = [start_idx] + list(perm)
                    if is_valid_order(group, test_perm, prev_balance):
                        return [group.index[i] for i in test_perm]
        # Fallback to any valid permutation
        for perm in permutations(indices):
            if is_valid_order(group, perm, prev_balance):
                return [group.index[i] for i in perm]
        return group.index  # Return original order if no valid permutation found

    # Reorder the DataFrame
    def reorder_dataframe(df):
        # Create a copy to avoid modifying the original
        df = df.copy()
        # Ensure timestamp is string
        df[df.columns[3]] = df[df.columns[3]].astype(str)
        # Sort by timestamp to ensure chronological order
        try:
            df = df.sort_values(by=df.columns[3], kind='mergesort')  # Stable sort
        except Exception as e:
            st.error(f"Error sorting by timestamp: {str(e)}")
            st.write("Timestamp values:", sorted(df[df.columns[3]].unique().tolist()))
            st.stop()
        
        # Initialize list to collect reordered indices
        reordered_indices = []
        
        # Track previous balances for each inventory + asset
        balance_tracker = {}
        
        # Log non-numeric assetUnitAdj or assetBalance
        non_numeric_rows = df[df[df.columns[11]].apply(lambda x: not isinstance(x, (int, float)) and not str(x).replace('.', '').replace('-', '').isdigit()) | 
                             df[df.columns[12]].apply(lambda x: not isinstance(x, (int, float)) and not str(x).replace('.', '').replace('-', '').isdigit())]
        if not non_numeric_rows.empty:
            st.warning(f"Found {len(non_numeric_rows)} rows with non-numeric assetUnitAdj or assetBalance. These will use original order.")
            st.write("Sample non-numeric rows:", non_numeric_rows[['timestamp', 'inventory', 'asset', 'assetUnitAdj', 'assetBalance']].head().to_dict())
        
        # Group by timestamp and inventory to ensure same inventory rows are together
        grouped = df.groupby([df.columns[3], df.columns[27]])  # timestamp, inventory
        
        for timestamp in sorted(df[df.columns[3]].unique()):
            # Get all rows for this timestamp
            timestamp_rows = df[df[df.columns[3]] == timestamp]
            # Process each inventory within this timestamp
            for inventory in sorted(timestamp_rows[df.columns[27]].unique()):
                try:
                    t_inv_group = grouped.get_group((timestamp, inventory))
                except KeyError:
                    st.warning(f"Group not found for timestamp {timestamp}, inventory {inventory}. Including rows in original order.")
                    t_inv_group = timestamp_rows[timestamp_rows[df.columns[27]] == inventory]
                    reordered_indices.extend(t_inv_group.index)
                    continue
                
                # Sort by original index to preserve non-buy/sell order
                t_inv_group = t_inv_group.sort_index()
                # Group by asset to handle balance logic
                asset_groups = t_inv_group.groupby(df.columns[9])  # asset
                
                for asset in sorted(t_inv_group[df.columns[9]].unique()):
                    try:
                        t_group = asset_groups.get_group(asset)
                    except KeyError:
                        st.warning(f"Asset group not found for {inventory}, {asset} at {timestamp}. Including rows in original order.")
                        t_group = t_inv_group[t_inv_group[df.columns[9]] == asset]
                        reordered_indices.extend(t_group.index)
                        continue
                    
                    # Split into buy/sell and non-buy/sell rows
                    buy_sell_rows = t_group[t_group["action"].str.lower().isin(["buy", "sell"])]
                    non_buy_sell_rows = t_group[~t_group["action"].str.lower().isin(["buy", "sell"])]
                    
                    # Get previous balance for this inventory + asset
                    prev_balance = balance_tracker.get((inventory, asset), None)
                    # Check if this is the earliest timestamp for this inventory + asset
                    is_earliest = timestamp == df[(df[df.columns[27]] == inventory) & (df[df.columns[9]] == asset)][df.columns[3]].min()
                    
                    if len(buy_sell_rows) > 1:
                        # Find valid order, prioritizing initial balance rows for earliest timestamp
                        valid_order = find_valid_permutation(buy_sell_rows, prev_balance, prioritize_initial=is_earliest)
                        buy_sell_ordered = buy_sell_rows.loc[valid_order]
                        # Update balance tracker with the last balance
                        try:
                            balance_tracker[(inventory, asset)] = float(buy_sell_ordered.iloc[-1][df.columns[12]])
                        except (ValueError, TypeError):
                            st.warning(f"Non-numeric assetBalance in last row for {inventory}, {asset} at {timestamp}. Skipping balance update.")
                    elif len(buy_sell_rows) == 1:
                        # Single buy/sell row: verify balance if prev_balance exists
                        row = buy_sell_rows.iloc[0]
                        try:
                            units = float(row[df.columns[11]])
                            current_balance = float(row[df.columns[12]])
                        except (ValueError, TypeError):
                            st.warning(f"Non-numeric assetUnitAdj or assetBalance for {inventory}, {asset} at {timestamp}. Using original order.")
                            buy_sell_ordered = buy_sell_rows
                            balance_tracker[(inventory, asset)] = None
                            continue
                        if prev_balance is not None:
                            expected_balance = prev_balance + units if row["action"].lower() == "buy" else prev_balance - units
                            if abs(expected_balance - current_balance) > 1e-6:
                                st.warning(f"Balance mismatch for {inventory}, {asset} at {timestamp}: expected {expected_balance}, got {current_balance}")
                        buy_sell_ordered = buy_sell_rows
                        balance_tracker[(inventory, asset)] = current_balance
                    else:
                        buy_sell_ordered = buy_sell_rows
                    
                    # Combine buy/sell and non-buy/sell rows
                    if not non_buy_sell_rows.empty:
                        combined = pd.concat([buy_sell_ordered, non_buy_sell_rows]).sort_index()
                        reordered_indices.extend(combined.index)
                    else:
                        reordered_indices.extend(buy_sell_ordered.index)
        
        # Include any remaining rows
        remaining_indices = df.index[~df.index.isin(reordered_indices)]
        if remaining_indices.size > 0:
            st.warning(f"Found {len(remaining_indices)} rows not processed. Including them in original order.")
            reordered_indices.extend(remaining_indices)
        
        # Ensure all rows are included
        if len(reordered_indices) != len(df):
            st.error(f"Row count mismatch: expected {len(df)} rows, got {len(reordered_indices)}.")
            st.write("Processed indices:", len(reordered_indices))
            st.stop()
        
        # Reorder the DataFrame
        reordered_df = df.loc[reordered_indices].reset_index(drop=True)
        
        # Verify chronological order
        try:
            if not reordered_df[df.columns[3]].astype(str).is_monotonic_increasing:
                st.warning("Output rows are not in strict chronological order. Please check the timestamp column.")
        except Exception as e:
            st.warning(f"Error verifying chronological order: {str(e)}")
        
        return reordered_df

    # Process the CSV
    try:
        reordered_df = reorder_dataframe(df)
        
        # Display reordered data
        st.write(f"Reordered CSV Data ({len(reordered_df)} rows):")
        st.dataframe(reordered_df)
        
        # Function to convert DataFrame to CSV and create download link
        def get_csv_download_link(df, filename="reordered_actions_report.csv"):
            csv_buffer = BytesIO()
            df.to_csv(csv_buffer, index=False)
            csv_buffer.seek(0)
            b64 = base64.b64encode(csv_buffer.read()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download Reordered CSV</a>'
            return href
        
        # Display download link
        st.markdown(get_csv_download_link(reordered_df), unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error processing the CSV: {str(e)}")
        st.write("Available columns:", list(df.columns))
        st.write("Please ensure the CSV has the required columns in the correct positions (timestamp in Column D, action in E, asset in J, assetUnitAdj in L, assetBalance in M, inventory in AC).")
