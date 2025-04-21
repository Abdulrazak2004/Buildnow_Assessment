import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# --- Page Configuration ---
# Set the layout to wide mode, add a title and icon
# The theme is now primarily controlled by .streamlit/config.toml
st.set_page_config(layout="wide", page_title="Buildnow PO Spend Analysis", page_icon="📊")

# --- Load Data ---
# Use caching to avoid reloading data on every interaction
@st.cache_data
def load_data(file_path):
    """Loads the categorized CSV data."""
    try:
        df = pd.read_csv(file_path, encoding='utf-8-sig') 
        # Basic data type conversion and cleaning
        df['Total Bcy'] = pd.to_numeric(df['Total Bcy'], errors='coerce')
        df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce') # Convert Quantity
        # Ensure IDs are treated as strings/objects for filtering
        if 'Project ID' in df.columns:
            df['Project ID'] = df['Project ID'].astype(str).fillna('N/A')
        if 'Purchase Order ID' in df.columns:
            df['Purchase Order ID'] = df['Purchase Order ID'].astype(str).fillna('N/A')

        df['Category'] = df['Category'].astype(str).fillna('Unknown')
        df['Item Name'] = df['Item Name'].astype(str).fillna('Unknown')

        # Drop rows where essential numeric conversions failed
        df.dropna(subset=['Total Bcy', 'Quantity'], inplace=True)
        return df
    except FileNotFoundError:
        st.error(f"Error: The data file was not found at {file_path}")
        st.warning("Please ensure you have run the categorization script first and the output CSV is in the correct location.")
        return None
    except KeyError as e:
        st.error(f"Error: Missing expected column in the CSV file: {e}. Please check the input file.")
        return None
    except Exception as e:
        st.error(f"Error loading or processing data: {e}")
        return None

# Define the path to your categorized data
DATA_FILE = "purchase_orders_categorized.csv" # Make sure this file exists
df_original = load_data(DATA_FILE)

# --- Helper Function for Formatting Numbers ---
def format_number(value, precision=0):
    """Formats a number with commas and specified precision."""
    return f"{value:,.{precision}f}"

# --- Main App Logic ---
if df_original is not None:

    st.title("📊 Buildnow Purchase Order Spend Analysis")
    st.markdown("Interactive dashboard to analyze spending patterns based on LLM-generated categories.")

    # --- Sidebar Filters ---
    st.sidebar.header("Filters")

    # Project ID Filter
    project_list = ['All Projects'] + sorted(df_original['Project ID'].unique().tolist())
    selected_projects = st.sidebar.multiselect(
        "Select Project ID(s):",
        options=project_list,
        default=['All Projects']
    )

    # Purchase Order ID Filter
    po_list = ['All Purchase Orders'] + sorted(df_original['Purchase Order ID'].unique().tolist())
    selected_po = st.sidebar.selectbox(
        "Select Purchase Order ID:",
        options=po_list,
        index=0 # Default to 'All Purchase Orders'
    )

    # Category Filter (Optional)
    category_list = ['All Categories'] + sorted(df_original['Category'].unique().tolist())
    selected_categories = st.sidebar.multiselect(
        "Select Category(s):",
        options=category_list,
        default=['All Categories']
    )

    # --- Filter Data Based on Selections ---
    df_filtered = df_original.copy()

    # Apply Project ID filter
    if 'All Projects' not in selected_projects and selected_projects:
        df_filtered = df_filtered[df_filtered['Project ID'].isin(selected_projects)]

    # Apply Purchase Order ID filter
    if selected_po != 'All Purchase Orders':
        df_filtered = df_filtered[df_filtered['Purchase Order ID'] == selected_po]

    # Apply Category filter
    if 'All Categories' not in selected_categories and selected_categories:
        df_filtered = df_filtered[df_filtered['Category'].isin(selected_categories)]

    # Check if filtering resulted in empty data
    if df_filtered.empty:
        st.warning("No data matches the selected filters.")
    else:
        # --- Row 1: Key Performance Indicators (KPIs) ---
        st.header("Overall Summary (Filtered)")
        kpi_cols = st.columns(5) # Create 5 columns for KPIs

        total_spend = df_filtered['Total Bcy'].sum()
        total_items = len(df_filtered)
        total_quantity = df_filtered['Quantity'].sum() # Calculate total quantity
        # Count unique categories excluding potential 'Error'/'Unknown' if desired
        num_categories = df_filtered[~df_filtered['Category'].isin(['Unknown', 'Error'])]['Category'].nunique()
        avg_spend = total_spend / total_items if total_items > 0 else 0

        kpi_cols[0].metric(label="Total Spend 💰", value=format_number(total_spend, 2))
        kpi_cols[1].metric(label="Total PO Items 🛒", value=format_number(total_items))
        kpi_cols[2].metric(label="Total Quantity 📦", value=format_number(total_quantity)) # Add Quantity KPI
        kpi_cols[3].metric(label="Unique Categories 🏷️", value=f"{num_categories}")
        kpi_cols[4].metric(label="Avg. Spend per Item 💸", value=format_number(avg_spend, 2))

        st.markdown("---") # Separator

        # --- Row 2: Spending & Quantity Distribution ---
        st.header("Spending & Quantity Distribution")
        dist_cols = st.columns(2) # Use 2 columns

        # Spending per Category Bar Chart
        spend_by_cat = df_filtered.groupby('Category')['Total Bcy'].sum().reset_index().sort_values(by='Total Bcy', ascending=False)
        fig_bar_spend = px.bar(
            spend_by_cat,
            x='Category',
            y='Total Bcy',
            title="Total Spend by Generated Category",
            labels={'Category': 'Category', 'Total Bcy': 'Total Spend'},
            height=400,
            template='plotly_dark' # <--- ADDED THIS
        )
        fig_bar_spend.update_layout(xaxis_title=None)
        dist_cols[0].plotly_chart(fig_bar_spend, use_container_width=True)

        # Quantity per Category Bar Chart (NEW)
        qty_by_cat = df_filtered.groupby('Category')['Quantity'].sum().reset_index().sort_values(by='Quantity', ascending=False)
        fig_bar_qty = px.bar(
            qty_by_cat,
            x='Category',
            y='Quantity',
            title="Total Quantity by Generated Category",
            labels={'Category': 'Category', 'Quantity': 'Total Quantity'},
            height=400,
            template='plotly_dark' # <--- ADDED THIS
        )
        fig_bar_qty.update_layout(xaxis_title=None)
        dist_cols[1].plotly_chart(fig_bar_qty, use_container_width=True)

        st.markdown("---")

        # --- Row 3: Detailed View (PO / Category) ---
        st.header("Detailed View")

        # If a specific PO is selected, show its items
        if selected_po != 'All Purchase Orders':
            st.subheader(f"Items in Purchase Order: {selected_po}")
            po_details = df_filtered[['Item Name', 'Category', 'Quantity', 'Total Bcy']].sort_values(by='Total Bcy', ascending=False)
            # Use st.dataframe for interactive tables. Adjust theme automatically.
            st.dataframe(po_details, use_container_width=True)
        else:
            # Otherwise, show the Category Deep Dive
            st.subheader("Category Deep Dive")
            available_categories = sorted(df_filtered['Category'].unique())
            if not available_categories:
                 st.warning("No categories found for deep dive with current filters.")
            else:
                # Dropdown to select category
                selected_category_deep_dive = st.selectbox(
                    "Select a Category to Explore:",
                    options=available_categories
                )

                if selected_category_deep_dive:
                    df_category_detail = df_filtered[df_filtered['Category'] == selected_category_deep_dive]

                    st.markdown(f"**Top Items in '{selected_category_deep_dive}' by Spend**")
                    # Display top items by spend in a table
                    top_items_table_spend = df_category_detail[['Item Name', 'Quantity', 'Total Bcy']].sort_values(by='Total Bcy', ascending=False).head(15)
                    st.dataframe(top_items_table_spend, use_container_width=True)

                    st.markdown(f"**Top Items in '{selected_category_deep_dive}' by Quantity**")
                     # Display top items by quantity in a table
                    top_items_table_qty = df_category_detail[['Item Name', 'Quantity', 'Total Bcy']].sort_values(by='Quantity', ascending=False).head(15)
                    st.dataframe(top_items_table_qty, use_container_width=True)


        st.markdown("---")

        # --- Row 4: Project-Specific Analysis (Only show if specific projects selected) ---
        if 'All Projects' not in selected_projects and selected_projects:
            st.header(f"Project-Specific Analysis for: {', '.join(selected_projects)}")
            proj_cols = st.columns(2)

            # Project Spend by Category Bar Chart
            project_spend_by_cat = df_filtered.groupby('Category')['Total Bcy'].sum().reset_index().sort_values(by='Total Bcy', ascending=False)
            fig_proj_bar = px.bar(
                project_spend_by_cat,
                x='Category',
                y='Total Bcy',
                title=f"Spend by Category within Selected Project(s)",
                labels={'Category': 'Category', 'Total Bcy': 'Total Spend'},
                height=400,
                template='plotly_dark' # <--- ADDED THIS
            )
            proj_cols[0].plotly_chart(fig_proj_bar, use_container_width=True)

            # Project Item Details Table
            top_project_items = df_filtered[['Item Name', 'Category', 'Quantity', 'Total Bcy']].sort_values(by='Total Bcy', ascending=False).head(15)
            proj_cols[1].subheader("Top Spending Items in Selected Project(s)")
            proj_cols[1].dataframe(top_project_items, use_container_width=True)

            st.markdown("---")


        # --- Row 5: Data Explorer ---
        with st.expander("Explore Raw Filtered Data"):
            # Dataframe theme will also adapt based on the config.toml
            st.dataframe(df_filtered, use_container_width=True)

else:
    # Display a message if data loading failed
    st.info("Dashboard cannot be displayed because the data failed to load.")
    st.info(f"Please ensure the file '{DATA_FILE}' exists in the same directory as the script.")