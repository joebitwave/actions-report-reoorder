import streamlit as st
import pandas as pd
import io
from datetime import datetime

# Set page configuration
st.set_page_config(page_title="Transaction Reordering App", layout="wide")

# Title
st.title("Transaction Reordering App")
st.markdown("Upload a CSV file to reorder transactions based on timestamp, inventory, asset, action, and asset balance.")

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
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

            # Ensure assetUnitAdj and assetBalance are numeric
            df['assetUnitAdj'] = pd.to_numeric(df['assetUnitAdj'], errors='coerce')
            df['assetBalance'] = pd.to_numeric(df['assetBalance'], errors='coerce')

            # Check for invalid data
            if df['timestamp'].isna().any():
                st.error("Invalid or missing timestamps in 'timestamp' column.")
            elif df['assetUnitAdj'].isna().any() or df['assetBalance'].isna().any():
                st.error("Invalid or missing numeric data in 'assetUnitAdj' or 'assetBalance' columns.")
            else:
                # Initialize progress bar
                st.write("Processing file...")
                progress_bar = st.progress(0)

                # Define custom sort order for 'action' column
                action_order = {
                    'buy': 1,
                    'sell': 2,
                    'fair-value-adjustment-upward': 3,
                    'fair-value-adjustment-downward': 4
                }
                # Map action values to their sort order; others get a high value to sort last
                df['action_order'] = df['action'].map(action_order).fillna(999).astype(int)

                # Sort the DataFrame
                # 1. timestamp (asc)
                # 2. inventory (asc)
                # 3. asset (asc)
                # 4. action (custom order: buy, sell, fair-value-adjustment-upward, fair-value-adjustment-downward)
                # 5. assetBalance (asc)
                df_sorted = df.sort_values(
                    by=['timestamp', 'inventory', 'asset', 'action_order', 'assetBalance'],
                    ascending=[True, True, True, True, True]
                )

                # Remove temporary column
                df_sorted = df_sorted.drop(columns=['action_order'])

                # Update progress to complete
                progress_bar.progress(1.0)

                # Display the reordered data
                st.subheader("Reordered Transactions")
                st.dataframe(df_sorted)

                # Generate downloadable CSV
                csv_buffer = io.StringIO()
                df_sorted.to_csv(csv_buffer, index=False)
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
