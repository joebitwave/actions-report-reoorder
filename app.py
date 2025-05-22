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
    st.write("Original CSV Data:")
    st.dataframe(df)
    
    # Validate required columns
    required_cols = [3, 4, 9, 11, 12, 27]  # Indices for timestamp, type, asset, assetUnitAdj, assetBalance, inventory
    if len(df.columns) < max(required_cols) + 1:
        st.error(f"CSV has too few columns. Expected at least {max(required_cols) + 1}, found {len(df.columns)}.")
        st.write("Available columns:", list(df.columns))
        st.stop()
    
    # Check if 'type' column exists
    if "type" not in df.columns:
        st.error("CSV is missing the 'type' column (expected in Column E).")
        st.write("Available columns:", list(df.columns))
        st.stop()
    
    # Function to validate balance transitions for a sequence of buy/sell transactions
    def is_valid_order(group, indices, prev_balance=None):
        balance = prev_balance if prev_balance is not None else group.iloc[0][df.columns[12]]  # assetBalance (M)
        for idx in indices:
            row = group.iloc[idx]
            action = row["type"].lower()  # type (E)
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
        # Sort by timestamp first
        df = df.sort_values(by=df.columns[3])  # timestamp (D)
        
        # Group by inventory (AC) and asset (J)
        grouped = df.groupby([df.columns[27], df.columns[9]])  # inventory and asset
        
        reordered_indices = []
        for _, group in grouped:
            # Further group by timestamp to handle ties
            timestamp_groups = group.groupby(df.columns[3])
            for _, t_group in timestamp_groups:
                # Split into buy/sell and non-buy/sell rows
                buy_sell_rows = t_group[t_group["type"].str.lower().isin(["buy", "sell"])]
                non_buy_sell_rows = t_group[~t_group["type"].str.lower().isin(["buy", "sell"])]
                
                if len(buy_sell_rows) > 1:
                    # Find valid order for buy/sell rows with tied timestamps
                    valid_order = find_valid_permutation(buy_sell_rows)
                    buy_sell_ordered = buy_sell_rows.loc[valid_order]
                else:
                    buy_sell_ordered = buy_sell_rows
                
                # Combine buy/sell and non-buy/sell rows, preserving non-buy/sell relative order
                if not non_buy_sell_rows.empty:
                    combined = pd.concat([buy_sell_ordered, non_buy_sell_rows]).sort_index()
                    reordered_indices.extend(combined.index)
                else:
                    reordered_indices.extend(buy_sell_ordered.index)
        
        # Reorder the DataFrame based on indices
        return df.loc[reordered_indices].reset_index(drop=True)

    # Process the CSV
    try:
        reordered_df = reorder_dataframe(df)
        
        # Display reordered data
        st.write("Reordered CSV Data:")
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
        st.write("Please ensure the CSV has the required columns in the correct positions (timestamp in Column D, type in E, asset in J, assetUnitAdj in L, assetBalance in M, inventory in AC).")
