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
                        # Sort action rows by timestamp
                        action_rows = action_rows.sort_values('timestamp')

                        # Recompute running balance
                        action_rows = action_rows.reset_index(drop=True)
                        if not action_rows.empty:
                            # Set first row's assetBalance to assetUnitAdj
                            action_rows.loc[0, 'assetBalance'] = action_rows.loc[0, 'assetUnitAdj']
                            # Compute subsequent balances
                            for i in range(1, len(action_rows)):
                                action_rows.loc[i, 'assetBalance'] = (
                                    action_rows.loc[i-1, 'assetBalance'] + action_rows.loc[i, 'assetUnitAdj']
                                )

                        # Handle same-timestamp rows
                        action_rows['temp_order'] = range(len(action_rows))  # Preserve balance order
                        action_rows = action_rows.sort_values(['timestamp', 'temp_order'])

                    # Append non-action rows (place last for each timestamp)
                    if not non_action_rows.empty:
                        non_action_rows = non_action_rows.sort_values('timestamp')
                        combined = pd.concat([action_rows, non_action_rows], ignore_index=True)
                    else:
                        combined = action_rows

                    # Remove temporary column if it exists
                    if 'temp_order' in combined.columns:
                        combined = combined.drop(columns=['temp_order'])

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
