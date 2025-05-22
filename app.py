import streamlit as st
import pandas as pd
import io
from datetime import datetime
import pytz
import numpy as np

# Set page configuration
st.set_page_config(page_title="Crypto Transaction Sorter", page_icon="📊", layout="wide")

# Title and description
st.title("Crypto Transaction Sorter")
st.markdown("""
Upload a CSV file containing cryptocurrency transactions to reorder the rows based on specific sorting rules.
The output file will have the same data, sorted by timestamp, inventory, asset, and maintaining running balance consistency.
Includes 'running_balance_recalc' and 'test' columns to verify balance accuracy.
""")

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        # Read the CSV file
        df = pd.read_csv(uploaded_file)

        # Define required columns and possible aliases for inventory
        required_columns = ['timestamp', 'action', 'asset', 'assetUnitAdj', 'assetBalance']
        inventory_aliases = ['inventory', 'Inventory', 'inventory_name', 'inv']

        # Check for required columns
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            st.error(f"Missing required columns: {', '.join(missing_columns)}")
        else:
            # Find inventory column
            inventory_col = None
            for alias in inventory_aliases:
                if alias in df.columns:
                    inventory_col = alias
                    break

            if inventory_col is None:
                st.error("No 'inventory' column found. Please specify the column name.")
                inventory_col = st.text_input("Enter the name of the inventory column:", "")
                if inventory_col and inventory_col in df.columns:
                    df = df.rename(columns={inventory_col: 'inventory'})
                else:
                    st.error("Specified inventory column not found in the CSV.")
                    st.stop()
            else:
                if inventory_col != 'inventory':
                    df = df.rename(columns={inventory_col: 'inventory'})

            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce', utc=True)
            if df['timestamp'].isna().any():
                st.error("Some timestamps could not be parsed. Please ensure all timestamps are valid.")
            else:
                # Function to sort buy/sell transactions globally for each inventory and asset
                def sort_buy_sell_global(df):
                    buy_sell = df[df['action'].isin(['buy', 'sell'])].copy()
                    non_buy_sell = df[~df['action'].isin(['buy', 'sell'])].copy()

                    if not buy_sell.empty:
                        sorted_dfs = []
                        # Group by inventory and asset
                        grouped = buy_sell.groupby(['inventory', 'asset'])

                        for (inv, asset), subgroup in grouped:
                            # Select the earliest transaction as the starting point
                            # Prefer "Beginning Balance" transactions if available
                            subgroup = subgroup.sort_values('timestamp')
                            if 'description' in subgroup.columns:
                                start_candidates = subgroup[subgroup['description'].str.contains('Beginning Balance', na=False)]
                                if not start_candidates.empty:
                                    start_row = start_candidates.iloc[[0]]
                                else:
                                    start_row = subgroup.iloc[[0]]
                            else:
                                start_row = subgroup.iloc[[0]]

                            result = start_row.copy()
                            remaining = subgroup.drop(start_row.index).copy()

                            # Iteratively build the sequence
                            while not remaining.empty:
                                last_balance = result.iloc[-1]['assetBalance']
                                # Find next row where assetBalance ≈ last_balance + assetUnitAdj
                                next_row = remaining[
                                    abs((remaining['assetUnitAdj'] + last_balance) - remaining['assetBalance']) < 1e-10
                                ]
                                if next_row.empty:
                                    # Fallback to closest match
                                    remaining['balance_diff'] = abs(
                                        (remaining['assetUnitAdj'] + last_balance) - remaining['assetBalance']
                                    )
                                    next_row = remaining.nsmallest(1, 'balance_diff')

                                # Add the first matching row
                                result = pd.concat([result, next_row.iloc[[0]]], ignore_index=True)
                                remaining = remaining.drop(next_row.index)

                            sorted_dfs.append(result.drop(columns=['balance_diff'] if 'balance_diff' in result.columns else []))

                        # Combine sorted buy/sell transactions
                        buy_sell_sorted = pd.concat(sorted_dfs, ignore_index=True)
                        return buy_sell_sorted, non_buy_sell
                    else:
                        return pd.DataFrame(), non_buy_sell

                # Sort buy/sell transactions globally
                buy_sell_sorted, non_buy_sell = sort_buy_sell_global(df)

                # Reintegrate with timestamps, placing non-buy/sell last
                def reassemble_with_timestamps(buy_sell, non_buy_sell):
                    if buy_sell.empty and non_buy_sell.empty:
                        return df
                    # Create a list to hold final sorted rows
                    final_dfs = []
                    # Group non-buy/sell by timestamp
                    non_buy_sell_groups = non_buy_sell.groupby('timestamp')
                    # Get unique timestamps
                    timestamps = sorted(df['timestamp'].unique())
                    for ts in timestamps:
                        # Get buy/sell transactions for this timestamp
                        ts_buy_sell = buy_sell[buy_sell['timestamp'] == ts].copy()
                        # Sort by inventory and asset
                        ts_buy_sell = ts_buy_sell.sort_values(['inventory', 'asset'])
                        # Get non-buy/sell for this timestamp
                        ts_non_buy_sell = non_buy_sell_groups.get_group(ts) if ts in non_buy_sell_groups.groups else pd.DataFrame()
                        # Sort non-buy/sell by inventory and asset
                        ts_non_buy_sell = ts_non_buy_sell.sort_values(['inventory', 'asset'])
                        # Combine, placing non-buy/sell last
                        ts_combined = pd.concat([ts_buy_sell, ts_non_buy_sell], ignore_index=True)
                        final_dfs.append(ts_combined)
                    return pd.concat(final_dfs, ignore_index=True)

                df_sorted = reassemble_with_timestamps(buy_sell_sorted, non_buy_sell)

                # Verify running balance consistency
                def verify_running_balance(df):
                    df['running_balance_recalc'] = np.nan
                    df['test'] = True
                    for (inv, asset), group in df[df['action'].isin(['buy', 'sell'])].groupby(['inventory', 'asset']):
                        indices = group.index
                        running_balance = 0
                        for i, idx in enumerate(indices):
                            row = df.loc[idx]
                            running_balance += row['assetUnitAdj']
                            df.loc[idx, 'running_balance_recalc'] = running_balance
                            df.loc[idx, 'test'] = abs(running_balance - row['assetBalance']) < 1e-10
                    return df

                df_sorted = verify_running_balance(df_sorted)

                # Check for any test = False, specifically highlighting Global- Asset Holdings and BTC
                inconsistent_rows = df_sorted[
                    (df_sorted['action'].isin(['buy', 'sell'])) & (~df_sorted['test']) &
                    (df_sorted['inventory'] == 'Global- Asset Holdings') & (df_sorted['asset'] == 'BTC')
                ]
                if not inconsistent_rows.empty:
                    st.warning(f"Inconsistent running balances found for Global- Asset Holdings and BTC. Check the 'test' column in the output CSV for details.")

                # Convert timestamp back to original format
                df_sorted['timestamp'] = df_sorted['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S %Z')

                # Display the sorted dataframe
                st.subheader("Sorted Transactions")
                st.dataframe(df_sorted)

                # Provide download link
                output = io.StringIO()
                df_sorted.to_csv(output, index=False)
                csv_data = output.getvalue()
                st.download_button(
                    label="Download Sorted CSV",
                    data=csv_data,
                    file_name=f"sorted_transactions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
else:
    st.info("Please upload a CSV file to proceed.")

# Sidebar with instructions
st.sidebar.header("Instructions")
st.sidebar.markdown("""
1. Upload a CSV file with transaction data.
2. The file must contain the following columns: `timestamp`, `action`, `asset`, `assetUnitAdj`, `assetBalance`, and an inventory column (e.g., `inventory`, `Inventory`).
3. If the inventory column is missing, enter its name in the provided input field.
4. The app will sort the transactions:
   - First by `timestamp` (earliest first).
   - Within each timestamp, by `inventory` and then `asset`.
   - For `buy` and `sell` actions, rows are ordered globally to maintain `assetBalance` as the sum of the previous balance and `assetUnitAdj`.
   - Non-`buy`/`sell` actions are placed last within each timestamp.
5. The output includes `running_balance_recalc` and `test` columns to verify balance consistency, with specific checks for `Global- Asset Holdings` and `BTC`.
6. Download the sorted CSV file.
""")
