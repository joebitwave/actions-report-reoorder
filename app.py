import streamlit as st
import pandas as pd
import io
from datetime import datetime
from itertools import permutations

# Set page configuration
st.set_page_config(page_title="Transaction Reordering App", layout="wide")

# Title
st.title("Transaction Reordering App")
st.markdown("Upload a CSV file to reorder transactions based on timestamp, inventory, asset, and running balance.")

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

def find_valid_permutation(rows, start_balance):
    """
    Find a permutation of rows where assetBalance = previous assetBalance + assetUnitAdj.
    Start with a row where assetBalance equals assetUnitAdj.
    """
    if not rows:
        return rows
    
    # Try each row as the starting point
    for start_idx in range(len(rows)):
        if abs(rows.iloc[start_idx]['assetBalance'] - rows.iloc[start_idx]['assetUnitAdj']) < 1e-10:
            start_row = rows.iloc[[start_idx]]
            remaining_rows = rows.drop(start_row.index)
            
            # Test permutations of remaining rows
            for perm in permutations(remaining_rows.index):
                ordered_rows = [start_row]
                current_balance = start_row['assetBalance'].iloc[0]
                
                # Check if permutation maintains balance
                valid = True
                for idx in perm:
                    row = remaining_rows.loc[[idx]]
                    expected_balance = current_balance + row['assetUnitAdj'].iloc[0]
                    if abs(row['assetBalance'].iloc[0] - expected_balance) > 1e-10:
                        valid = False
                        break
                    ordered_rows.append(row)
                    current_balance = row['assetBalance'].iloc[0]
                
                if valid:
                    return pd.concat(ordered_rows, ignore_index=True)
    
    return None  # No valid permutation found

if uploaded_file is not None:
    try:
        # Read the uploaded CSV file
        df = pd.read_csv(uploaded_file)

        # Verify required columns exist
        required_columns = ['timestamp', 'action', 'asset', 'assetUnitAdj', 'assetBalance', 'inventory']
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            st.error(f"Missing required columns: {', '.join(missing_cols)}")
        else:
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])

            # Ensure assetUnitAdj and assetBalance are numeric
            df['assetUnitAdj'] = pd.to_numeric(df['assetUnitAdj'], errors='coerce')
            df['assetBalance'] = pd.to_numeric(df['assetBalance'], errors='coerce')

            # Check for invalid numeric data
            if df['assetUnitAdj'].isna().any() or df['assetBalance'].isna().any():
                st.error("Invalid or missing numeric data in 'assetUnitAdj' or 'assetBalance' columns.")
            else:
                # Initialize output DataFrame
                output_df = pd.DataFrame()

                # Group by inventory and asset to process each combination
                for (inventory, asset), group in df.groupby(['inventory', 'asset']):
                    # Split into action rows (buy/sell) and non-action rows
                    action_rows = group[group['action'].isin(['buy', 'sell'])].copy()
                    non_action_rows = group[~group['action'].isin(['buy', 'sell'])].copy()

                    if not action_rows.empty:
                        # Group action rows by timestamp
                        action_rows = action_rows.sort_values('timestamp')
                        grouped_by_ts = action_rows.groupby('timestamp')
                        ordered_action_rows = []

                        for ts, ts_group in grouped_by_ts:
                            # Find valid permutation for this timestamp
                            valid_order = find_valid_permutation(ts_group, None)
                            if valid_order is None:
                                st.error(f"No valid order found for asset '{asset}' and inventory '{inventory}' at timestamp '{ts}'.")
                                raise ValueError(f"Cannot satisfy assetBalance condition for {asset} in {inventory} at {ts}")
                            ordered_action_rows.append(valid_order)

                        # Concatenate ordered action rows
                        action_rows = pd.concat(ordered_action_rows, ignore_index=True) if ordered_action_rows else pd.DataFrame()

                    # Append non-action rows (place last for each timestamp)
                    if not non_action_rows.empty:
                        non_action_rows = non_action_rows.sort_values('timestamp')
                        combined = pd.concat([action_rows, non_action_rows], ignore_index=True)
                    else:
                        combined = action_rows

                    # Append to output DataFrame
                    output_df = pd.concat([output_df, combined], ignore_index=True)

                # Final sort: timestamp, inventory, asset
                output_df = output_df.sort_values(['timestamp', 'inventory', 'asset'])

                # Display the reordered data
                st.subheader("Reordered Transactions")
                st.dataframe(output_df)

                # Generate downloadable CSV
                csv_buffer = io.StringIO()
                output_df.to_csv(csv_buffer, index=False)
                csv_data = csv_buffer.getvalue()

                st.download_button(
                    label="Download Reordered CSV",
                    data=csv_data,
                    file_name=f"reordered_transactions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
else:
    st.info("Please upload a CSV file to proceed.")

# Footer
st.markdown("---")
st.write("Built with Streamlit | © 2025 Your Project")
