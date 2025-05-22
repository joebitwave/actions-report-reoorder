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
    df = pd.read_csv(uploaded_file)
    
    # Display original data
    st.write("Original CSV Data:")
    st.dataframe(df)
    
    # Function to validate balance transitions for a sequence of buy/sell transactions
    def is_valid_order(group, indices, prev_balance=None):
        balance = prev_balance if prev_balance is not None else group.iloc[0]["runningBalance"]
        for idx in indices:
            row = group.iloc[idx]
            action = row["Action Type"].lower()
            units = row["assetUnitAdj"]
            current_balance = row["runningBalance"]
            
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
        # Sort by Timestamp first
        df = df.sort_values(by=df.columns[3])  # Column D (Timestamp)
        
        # Group by Inventory ID (Column AC) and Asset (Column J)
        grouped = df.groupby([df.columns[27], df.columns[9]])  # AC and J
        
        reordered_indices = []
        for _, group in grouped:
            # Further group by Timestamp to handle ties
            timestamp_groups = group.groupby(df.columns[3])
            for _, t_group in timestamp_groups:
                # Split into buy/sell and non-buy/sell rows
                buy_sell_rows = t_group[t_group["Action Type"].str.lower().isin(["buy", "sell"])]
                non_buy_sell_rows = t_group[~t_group["Action Type"].str.lower().isin(["buy", "sell"])]
                
                if len(buy_sell_rows) > 1:
                    # Find valid order for buy/sell rows with tied timestamps
                    valid_order = find_valid_permutation(buy_sell_rows)
                    buy_sell_ordered = buy_sell_rows.loc[valid_order]
                else:
                    buy_sell_ordered = buy_sell_rows
                
                # Combine buy/sell and non-buy/sell rows, preserving non-buy/sell relative order
                if not non_buy_sell_rows.empty:
                    # Merge while maintaining original relative order of non-buy/sell rows
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
