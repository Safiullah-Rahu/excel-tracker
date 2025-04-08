import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
import io

st.set_page_config(layout="wide", page_title="Sales Performance Tracker")

def process_data(df):
    """Process the uploaded data to extract necessary information."""
    # Convert relevant columns to numeric
    numeric_columns = ['Quantity(EA)', 'Gross Amount', 'Net Amount', 'Sales Tax', 'Value for GST']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Extract route information
    route_data = df.groupby(['Route Code', 'Route Name']).agg({
        'Customer Code': 'nunique',
        'Quantity(EA)': 'sum',
        'Net Amount': 'sum'
    }).reset_index()
    
    route_data.rename(columns={
        'Customer Code': 'CUSTOMERS',
        'Quantity(EA)': 'PROD',
        'Net Amount': 'MSL'
    }, inplace=True)
    
    # Calculate percentage for products - handle NaN values
    total_prod = route_data['PROD'].sum()
    if total_prod > 0:
        route_data['PROD (%)'] = (route_data['PROD'] / total_prod * 100).round(0).fillna(0).astype(int)
    else:
        route_data['PROD (%)'] = 0
    
    # Process brand data
    brand_data = df.groupby(['Route Code', 'Route Name', 'Brand']).agg({
        'Customer Code': 'nunique',
        'Quantity(EA)': 'sum',
        'Net Amount': 'sum'
    }).reset_index()
    
    brand_data.rename(columns={
        'Customer Code': 'CUSTOMERS',
        'Quantity(EA)': 'PROD',
        'Net Amount': 'MSL'
    }, inplace=True)
    
    # Calculate percentage for brand products - handling NaN values
    brand_totals = brand_data.groupby('Brand')['PROD'].transform('sum')
    # Calculate percentages as Series (not ndarray) to allow pandas methods
    percentages = (brand_data['PROD'] / brand_totals * 100).round(0)
    # Replace infinities and NaNs with zeros
    brand_data['PROD (%)'] = percentages.replace([np.inf, -np.inf, np.nan], 0).astype(int)
    
    # Get unique brands
    brands = df['Brand'].unique()
    
    # Create a pivot table for brand performance by route
    brand_pivot = pd.pivot_table(
        brand_data, 
        values=['PROD', 'PROD (%)'], 
        index=['Route Code', 'Route Name'],
        columns=['Brand'],
        aggfunc='sum',
        fill_value=0
    ).reset_index()
    
    # Flatten multi-level columns
    brand_pivot.columns = [' '.join(col).strip() if isinstance(col, tuple) else col for col in brand_pivot.columns]
    
    # Return processed data
    return route_data, brand_data, brands, brand_pivot

def create_brand_summary(df):
    """Create summary of products by category."""
    # Extract primary brands from the data
    # Based on the image, we'll focus on Dettol Soap, Harpic, Mortein, and Veet
    
    # Get the product categories
    brand_mapping = {
        'BR03-Dettol': {
            'SG02-Dettol Soap': 'DETTOL SOAP',
            'SG03-Dettol Hand Wash': 'DETTOL HAND WASH'
        },
        'BR15-Veet': 'VEET'
    }
    
    # Create a new column for product category
    def map_category(row):
        brand = row['Brand']
        segment = row['Segment2']
        
        if brand in brand_mapping:
            if isinstance(brand_mapping[brand], dict):
                if segment in brand_mapping[brand]:
                    return brand_mapping[brand][segment]
                else:
                    return brand.split('-')[1].upper()
            else:
                return brand_mapping[brand]
        else:
            # Default to brand name if no mapping exists
            return brand.split('-')[1].upper() if '-' in brand else brand.upper()
    
    df['Product Category'] = df.apply(map_category, axis=1)
    
    # Group by route and product category
    category_data = df.groupby(['Route Code', 'Route Name', 'Product Category']).agg({
        'Customer Code': 'nunique',
        'Quantity(EA)': 'sum',
        'Net Amount': 'sum'
    }).reset_index()
    
    category_data.rename(columns={
        'Customer Code': 'CUSTOMERS',
        'Quantity(EA)': 'PROD',
        'Net Amount': 'MSL'
    }, inplace=True)
    
    # Calculate percentage for category products - handling NaN values
    category_totals = category_data.groupby('Product Category')['PROD'].transform('sum')
    
    # Calculate percentages as Series (not ndarray) to allow pandas methods
    percentages = (category_data['PROD'] / category_totals * 100).round(0)
    # Replace infinities and NaNs with zeros
    category_data['PROD (%)'] = percentages.replace([np.inf, -np.inf, np.nan], 0).astype(int)
    
    # Create a pivot table for category performance by route
    category_pivot = pd.pivot_table(
        category_data, 
        values=['PROD', 'PROD (%)'], 
        index=['Route Code', 'Route Name'],
        columns=['Product Category'],
        aggfunc='sum',
        fill_value=0
    ).reset_index()
    
    # Flatten multi-level columns
    category_pivot.columns = [' '.join(col).strip() if isinstance(col, tuple) else col for col in category_pivot.columns]
    
    return category_pivot, category_data

def display_performance_view(route_data, category_pivot):
    """Display the Performance | OB Wise tab."""
    st.header("Performance | OB Wise")
    
    # Display the top metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Customers", f"{route_data['CUSTOMERS'].sum()}")
    with col2:
        st.metric("Total Products", f"{route_data['PROD'].sum()}")
    with col3:
        st.metric("Total MSL", f"₹{route_data['MSL'].sum():,.2f}")
    
    # Display the main performance table
    st.subheader("Route Performance Overview")
    
    # Create a dataframe for the first table in the image
    performance_df = route_data[['Route Code', 'Route Name', 'CUSTOMERS', 'PROD', 'PROD (%)']]
    
    # Add grand total
    grand_total = pd.DataFrame({
        'Route Code': ['GRAND TOTAL'],
        'Route Name': [''],
        'CUSTOMERS': [performance_df['CUSTOMERS'].sum()],
        'PROD': [performance_df['PROD'].sum()],
        'PROD (%)': [performance_df['PROD (%)'].mean().round(0).astype(int)]
    })
    
    performance_df = pd.concat([performance_df, grand_total])
    
    # Display the table with formatting - use Streamlit's native styling options
    st.dataframe(
        performance_df,
        use_container_width=True,
        column_config={
            "PROD (%)": st.column_config.ProgressColumn(
                "PROD (%)",
                help="Product percentage by route",
                format="%d%%",
                min_value=0,
                max_value=100,
            ),
        },
        hide_index=True
    )
    
    # Display the brand performance table
    st.subheader("Brand Performance by Route")
    st.dataframe(category_pivot, use_container_width=True, hide_index=True)

def display_summary_view(df):
    """Display the Summary | Productivity & MSL tab."""
    st.header("Summary | Productivity & MSL")
    
    # Calculate summary metrics
    total_customers = df['Customer Code'].nunique()
    total_products = df['Quantity(EA)'].sum()
    total_msl = df['Net Amount'].sum()
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Customers", f"{total_customers}")
    with col2:
        st.metric("Total Products", f"{total_products}")
    with col3:
        st.metric("Total MSL", f"₹{total_msl:,.2f}")
    
    # Create a summary chart
    brand_summary = df.groupby('Brand').agg({
        'Quantity(EA)': 'sum',
        'Net Amount': 'sum',
        'Customer Code': 'nunique'
    }).reset_index()
    
    # Create a bar chart for products by brand
    fig = px.bar(brand_summary, x='Brand', y='Quantity(EA)', 
                 text=brand_summary['Quantity(EA)'].round(0).astype(int), 
                 title='Product Distribution by Brand',
                 labels={'Quantity(EA)': 'Products', 'Brand': 'Brand'})
    
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Create a summary table
    st.subheader("Brand Performance Summary")
    
    summary_df = brand_summary.copy()
    summary_df.columns = ['Brand', 'Products', 'Net Amount', 'Customers']
    
    # Handle division by zero for percentages
    total_products = summary_df['Products'].sum()
    total_net_amount = summary_df['Net Amount'].sum()
    
    # Calculate product percentages
    if total_products > 0:
        summary_df['Products %'] = (summary_df['Products'] / total_products * 100).round(1)
    else:
        summary_df['Products %'] = 0.0
        
    # Calculate net amount percentages
    if total_net_amount > 0:
        summary_df['Net Amount %'] = (summary_df['Net Amount'] / total_net_amount * 100).round(1)
    else:
        summary_df['Net Amount %'] = 0.0
    
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

def main():
    st.title("Sales Performance Tracker")
    
    # File upload widget
    st.sidebar.header("Upload Data")
    uploaded_file = st.sidebar.file_uploader("Upload Excel file with sales data", type=["xlsx", "xls", "csv"])
    
    # Sample data option
    use_sample_data = st.sidebar.checkbox("Use sample data", value=False)
    
    # Process data only if a file is uploaded or sample data is selected
    if uploaded_file is not None or use_sample_data:
        try:
            # Load data
            if uploaded_file is not None:
                # Determine file type and read accordingly
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                st.sidebar.success(f"Successfully loaded data from {uploaded_file.name}")
            else:
                # Use sample data from a demo file or embedded data
                st.sidebar.info("Using sample data for demonstration")
                # Create a sample dataframe with structure similar to the expected input
                # This is a simplified version for demonstration
                sample_data = {
                    'Year': [2025] * 10,
                    'Month': [4] * 10,
                    'Distributor': ['34140860'] * 10,
                    'Area': ['Hyderabad'] * 10,
                    'Town': ['Hyderabad'] * 10,
                    'Distributor Name': ['PREMIER SALES (PRIVATE) LIMITED HYD'] * 10,
                    'Customer Code': ['HYD0' + str(i) for i in range(10)],
                    'Customer Name': ['Customer ' + str(i) for i in range(10)],
                    'Route Code': ['PHDRC' + str(i % 5 + 16) for i in range(10)],
                    'Route Name': ['Zone ' + chr(65 + i % 5) + ' - Route ' + str(i % 5) for i in range(10)],
                    'Brand': ['BR03-Dettol', 'BR15-Veet', 'BR03-Dettol', 'BR15-Veet', 'BR03-Dettol', 
                             'BR03-Dettol', 'BR15-Veet', 'BR03-Dettol', 'BR15-Veet', 'BR03-Dettol'],
                    'Segment2': ['SG02-Dettol Soap', 'SG07-Veet Cream', 'SG03-Dettol Hand Wash', 'SG08-Veet Lotion', 'SG02-Dettol Soap',
                                'SG02-Dettol Soap', 'SG07-Veet Cream', 'SG03-Dettol Hand Wash', 'SG08-Veet Lotion', 'SG02-Dettol Soap'],
                    'Quantity(EA)': [3, 2, 1, 4, 5, 2, 3, 4, 1, 2],
                    'Gross Amount': [300, 400, 200, 800, 500, 200, 300, 400, 200, 300],
                    'Net Amount': [350, 450, 220, 880, 550, 220, 330, 440, 220, 330],
                    'Value for GST': [330, 430, 210, 850, 530, 210, 320, 420, 210, 320]
                }
                df = pd.DataFrame(sample_data)
            
            # Process the data
            route_data, brand_data, brands, brand_pivot = process_data(df)
            category_pivot, category_data = create_brand_summary(df)
            
            # Create tabs for different views
            tabs = st.tabs([
                "Performance | OB Wise", 
                "Channelwise", 
                "MTD", 
                "Summary | Productivity & MSL", 
                "Daily Productivity | Harpic", 
                "Booking Plan"
            ])
            
            # Populate the tabs
            with tabs[0]:
                display_performance_view(route_data, category_pivot)
            
            with tabs[3]:  # Summary | Productivity & MSL
                display_summary_view(df)
            
            # Placeholder for other tabs
            for i, tab_name in enumerate(["Channelwise", "MTD", "Daily Productivity | Harpic", "Booking Plan"]):
                if i != 0 and i != 3:  # Skip tabs we've already implemented
                    with tabs[i if i < 3 else i+1 if i == 3 else i]:
                        st.header(tab_name)
                        st.info(f"This tab is under development. The {tab_name} view will be available soon.")
            
        except Exception as e:
            st.error(f"Error processing data: {e}")
            st.exception(e)
    else:
        # Instructions when no file is uploaded
        st.info("Please upload an Excel file or use sample data to get started.")
        st.markdown("""
        ## Expected Data Format
        
        The Excel file should contain the following columns:
        
        - Year, Month, Distributor, Area, Town
        - Customer Code, Customer Name
        - Route Code, Route Name
        - Brand, Segment2 (product segment)
        - Quantity(EA) (quantity in equivalent units)
        - Net Amount (sales value)
        - And other related sales data
        
        The application will process this data to create a sales performance tracker similar to the one shown in the screenshots.
        """)

if __name__ == "__main__":
    main()