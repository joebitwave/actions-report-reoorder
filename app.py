import streamlit as st
import pandas as pd
import io
from datetime import datetime
import pytz

# Set page configuration
st.set_page_config(page_title="Crypto Transaction Sorter", page_icon="📊", layout="wide")

# Title and description
st.title("Crypto Transaction Sorter")
st.markdown("""
Upload a CSV file containing cryptocurrency transactions to reorder the rows based on specific sorting rules.
The output file will have the same data, sorted by timestamp, inventory, asset, and maintaining running balance consistency.
""")

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        # Read the CSV file
        df = pd.read_csv(uploaded_file)

        # Validate required columns
        required_columns = ['timestamp', 'action', 'asset', 'assetUnitAdj', 'assetBalance', 'inventory']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            st.error(f"Missing required columns: {', '.join(missing_columns)}")
        else:
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce', utc=True)
            if df['timestamp'].isna().any():
                st.error("Some timestamps could not be parsed. Please ensure all timestamps are valid.")
            else:
                # Function to sort within timestamp groups
                def sort_within_timestamp(group):
                    # Separate buy/sell and non-buy/sell actions
                    buy_sell = group[group['action'].isin(['buy', 'sell'])].copy()
                    non_buy_sell = group[~group['action'].isin(['buy', 'sell'])].copy()

                    if not buy_sell.empty:
                        # Identify starting points (where assetUnitAdj == assetBalance)
                        buy_sell['is_start'] = buy_sell['assetUnitAdj'] == buy_sell['assetBalance']
                        
                        # Group by inventory and asset
                        grouped = buy_sell.groupby(['inventory', 'asset'])
                        sorted_dfs = []

                        for (inv, asset), subgroup in grouped:
                            # Get the starting point (first transaction)
                            start_row = subgroup[subgroup['is_start']]
                            if start_row.empty:
                                st.warning(f"No starting point found for inventory {inv} and asset {asset}. Using earliest transaction.")
                                start_row = subgroup.nsmallest(1, 'assetBalance')
                            
                            # Initialize result with starting point
                            result = start_row.copy()
                            remaining = subgroup.drop(start_row.index).copy()

                            # Iteratively add rows ensuring running balance matches
                            while not remaining.empty:
                                last_balance = result.iloc[-1]['assetBalance']
                                # Find next row where assetUnitAdj + last_balance equals assetBalance
                                next_row = remaining[
                                    (remaining['assetUnitAdj'] + last_balance).round(8) == remaining['assetBalance'].round(8)
                                ]
                                if next_row.empty:
                                    st.warning(f"Could not find consistent running balance for inventory {inv}, asset {asset}.")
                                    # Fall back to smallest balance difference
                                    remaining['balance_diff'] = abs(
                                        (remaining['assetUnitAdj'] + last_balance) - remaining['assetBalance']
                                    )
                                    next_row = remaining.nsmallest(1, 'balance_diff')
                                    remaining = remaining.drop(next_row.index)
                                    result = pd.concat([result, next_row], ignore_index=True)
                                    continue
                                
                                # Add the first matching row
                                result = pd.concat([result, next_row.iloc[[0]]], ignore_index=True)
                                remaining = remaining.drop(next_row.index)

                            sorted_dfs.append(result.drop(columns=['is_start'] if 'is_start' in result.columns else []))

                        # Combine sorted buy/sell transactions
                        buy_sell_sorted = pd.concat(sorted_dfs, ignore_index=True) if sorted_dfs else pd.DataFrame()

                        # Sort non-buy/sell by inventory and asset
                        non_buy_sell = non_buy_sell.sort_values(['inventory', 'asset'])

                        # Combine buy/sell and non-buy/sell
                        return pd.concat([buy_sell_sorted, non_buy_sell], ignore_index=True)
                    else:
                        # Only non-buy/sell actions, sort by inventory and asset
                        return non_buy_sell.sort_values(['inventory', 'asset'])

                # Sort by timestamp and apply within-timestamp sorting
                df_sorted = df.sort_values('timestamp').groupby('timestamp').apply(sort_within_timestamp).reset_index(drop=True)

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
2. The file must contain the following columns: `timestamp`, `action`, `asset`, `assetUnitAdj`, `assetBalance`, `inventory`.
3. The app will sort the transactions:
   - First by `timestamp` (earliest first).
   - Within each timestamp, by `inventory` and then `asset`.
   - For `buy` and `sell` actions, rows are ordered to maintain `assetBalance` as the sum of the previous balance and `assetUnitAdj`.
   - Non-`buy`/`sell` actions are placed last within each timestamp.
4. Download the sorted CSV file.
""")
