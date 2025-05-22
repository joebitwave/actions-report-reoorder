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
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error reading CSV: {str(e)}")
        st.write("Please ensure the CSV is properly formatted with headers.")
        st.stop()
    
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
        balance = prev_balance if prev_balance is not None else group.iloc[0][df.columns[12]]  # assetBalance (M)
        for idx in indices:
            row = group.iloc[idx]
            action = row["action"].lower()  # action (E)
            units = row[df.columns[11]]  # assetUnitAdj (L)
            current_balance = row[df.columns[12]]  # assetBalance (M)
            
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
    def find_valid_permutation(group):
        indices = list(range(len(group)))
        for perm in permutations(indices):
            if is_valid_order(group, perm):
                return [group.index[i] for i in perm]
        return group.index  # Return original order if no valid permutation found

    # Reorder the DataFrame
    def reorder_dataframe(df):
        # Create a copy to avoid modifying the original
        df = df.copy()
        # Sort by timestamp to ensure chronological order
        df = df.sort_values(by=df.columns[3], kind='mergesort')  # Stable sort
        
        # Initialize list to collect reordered indices
        reordered_indices = []
        # Group by inventory, asset, and timestamp to identify tied buy/sell rows
        grouped = df.groupby([df.columns[27], df.columns[9], df.columns[3]])  # inventory, asset, timestamp
        
        # Iterate over sorted timestamps to maintain chronological order
        for timestamp in sorted(df[df.columns[3]].unique()):
            # Get all rows for this timestamp
            timestamp_rows = df[df[df.columns[3]] == timestamp]
            # Process each inventory + asset group within this timestamp
            for (inv, asset), group in timestamp_rows.groupby([df.columns[27], df.columns[9]]):
                # Get the corresponding group from the grouped object
                try:
                    t_group = grouped.get_group((inv, asset, timestamp))
                except KeyError:
                    continue  # Skip if group not found (shouldn't happen)
                
                # Sort by original index to preserve non-buy/sell order
                t_group = t_group.sort_index()
                # Split into buy/sell and non-buy/sell rows
                buy_sell_rows = t_group[t_group["action"].str.lower().isin(["buy", "sell"])]
                non_buy_sell_rows = t_group[~t_group["action"].str.lower().isin(["buy", "sell"])]
                
                if len(buy_sell_rows) > 1:
                    # Find valid order for buy/sell rows with tied timestamps
                    valid_order = find_valid_permutation(buy_sell_rows)
                    buy_sell_ordered = buy_sell_rows.loc[valid_order]
                else:
                    buy_sell_ordered = buy_sell_rows
                
                # Combine buy/sell and non-buy/sell rows
                if not non_buy_sell_rows.empty:
                    combined = pd.concat([buy_sell_ordered, non_buy_sell_rows]).sort_index()
                    reordered_indices.extend(combined.index)
                else:
                    reordered_indices.extend(buy_sell_ordered.index)
        
        # Include any remaining rows (e.g., missing inventory/asset)
        remaining_indices = df.index[~df.index.isin(reordered_indices)]
        if remaining_indices.size > 0:
            st.warning(f"Found {len(remaining_indices)} rows with missing inventory or asset values. Including them in chronological order.")
            reordered_indices.extend(remaining_indices)
        
        # Ensure all rows are included
        if len(reordered_indices) != len(df):
            st.error(f"Row count mismatch: expected {len(df)} rows, got {len(reordered_indices)}.")
            st.stop()
        
        # Reorder the DataFrame based on indices and ensure chronological order
        reordered_df = df.loc[reordered_indices].sort_values(by=df.columns[3], kind='mergesort').reset_index(drop=True)
        
        # Verify chronological order
        if not reordered_df[df.columns[3]].is_monotonic_increasing:
            st.warning("Output rows are not in strict chronological order. Please check the timestamp column.")
        
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
