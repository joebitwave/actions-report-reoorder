import streamlit as st
import pandas as pd
import io
from datetime import datetime

# Set page configuration
st.set_page_config(page_title="Transaction Reordering App", layout="wide")

# Title
st.title("Transaction Reordering App")
st.markdown("Upload a CSV file to reorder transactions based on timestamp, inventory, asset, and running balance.")

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

def order_rows_by_balance(rows):
    """
    Order rows to satisfy assetBalance = previous assetBalance + assetUnitAdj.
    Try each row as the starting point and greedily select subsequent rows.
    Returns None if no valid order is found.
    """
    if rows.empty:
        return None
    
    # Try each row as the starting point
    for start_idx in range(len(rows)):
        start_row = rows.iloc[[start_idx]]
        ordered_rows = [start_row]
        current_balance = start_row['assetBalance'].iloc[0]
        remaining_rows = rows.drop(start_row.index)
        
        # Greedily select rows that match the expected balance
        while not remaining_rows.empty:
            next_row = remaining_rows[abs(remaining_rows['assetBalance'] - (current_balance + remaining_rows['assetUnitAdj'])) < 1e-10]
            if next_row.empty:
                break  # No valid next row, try a different starting point
            # Take the first matching row
            next_row = next_row.iloc[[0]]
            ordered_rows.append(next_row)
            current_balance = next_row['assetBalance'].iloc[0]
            remaining_rows = remaining_rows.drop(next_row.index)
        
        # If all rows are used, return the ordered sequence
        if remaining_rows.empty:
            return pd.concat(ordered_rows, ignore_index=True)
    
    return None  # No valid order found

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
                # Initialize progress bar and warnings
                st.write("Processing file...")
                progress_bar = st.progress(0)
                warnings = []
                total_groups = len(df.groupby(['inventory', 'asset']))
                processed_groups = 0

                # Initialize output DataFrame
                output_df = pd.DataFrame()

                # Group by inventory and asset to process each combination
                for (inventory, asset), group in df.groupby(['inventory', 'asset']):
                    # Update progress
                    processed_groups += 1
                    progress_bar.progress(min(processed_groups / total_groups, 1.0))
                    st.write(f"Processing asset '{asset}' in inventory '{inventory}'...")

                    # Split into action rows (buy/sell) and non-action rows
                    action_rows = group[group['action'].isin(['buy', 'sell'])].copy()
                    non_action_rows = group[~group['action'].isin(['buy', 'sell'])].copy()

                    if not action_rows.empty:
                        # Group action rows by timestamp
                        action_rows = action_rows.sort_values('timestamp')
                        grouped_by_ts = action_rows.groupby('timestamp')
                        ordered_action_rows = []

                        for ts, ts_group in grouped_by_ts:
                            # Warn if group is large
                            if len(ts_group) > 10:
                                st.warning(f"Large group ({len(ts_group)} rows) for asset '{asset}' and inventory '{inventory}' at timestamp '{ts}'. Processing may be slow.")

                            # Order rows by balance condition
                            valid_order = order_rows_by_balance(ts_group)
                            if valid_order is None:
                                warning_msg = f"Warning: No valid order found for asset '{asset}' and inventory '{inventory}' at timestamp '{ts}'. Rows:\n{ts_group[['timestamp', 'action', 'assetUnitAdj', 'assetBalance']].to_string()}\nThese rows will be appended unsorted."
                                warnings.append(warning_msg)
                                st.warning(warning_msg)
                                ordered_action_rows.append(ts_group)  # Append unsorted
                            else:
                                ordered_action_rows.append(valid_order)

                        # Concatenate ordered action rows
                        action_rows = pd.concat(ordered_action_rows, ignore_index=True) if ordered_action_rows else pd.DataFrame(columns=action_rows.columns)

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

                # Display warnings if any
                if warnings:
                    st.warning("Some groups could not be ordered correctly. See details above. Output may be incomplete.")

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

                st.success("Processing complete!")

    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
else:
    st.info("Please upload a CSV file to proceed.")

# Footer
st.markdown("---")
st.write("Built with Streamlit | © 2025 Your Project")
