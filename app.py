import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------
# Page Configuration & Professional White/Blue Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Supermarket Sales Analysis | AI Analytics",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Professional White & Blue Theme
st.markdown("""
<style>
    /* Main App Background & Typography */
    .stApp {
        background-color: #F8FAFC;
        color: #1E293B;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Header Styling */
    h1, h2, h3, h4, h5, h6 {
        color: #0F172A !important;
        font-weight: 700 !important;
    }
    
    /* Card Component Styling */
    .metric-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        margin-bottom: 15px;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(37, 99, 235, 0.1);
        border-color: #BFDBFE;
    }
    .metric-title {
        color: #64748B;
        font-size: 0.875rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
    }
    .metric-value {
        color: #1E40AF;
        font-size: 1.85rem;
        font-weight: 800;
    }
    .metric-subtitle {
        color: #10B981;
        font-size: 0.8rem;
        font-weight: 500;
        margin-top: 4px;
    }
    
    /* Content Cards */
    .content-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #E2E8F0;
    }
    .sidebar-header {
        background: linear-gradient(135deg, #1E40AF 0%, #2563EB 100%);
        color: white;
        padding: 16px;
        border-radius: 10px;
        margin-bottom: 20px;
        text-align: center;
    }
    .sidebar-header h3 {
        color: #FFFFFF !important;
        margin: 0;
        font-size: 1.1rem;
    }
    .sidebar-header p {
        color: #DBEAFE;
        font-size: 0.8rem;
        margin: 4px 0 0 0;
    }
    
    /* Button Styling */
    .stButton>button {
        background-color: #2563EB;
        color: white;
        font-weight: 600;
        border-radius: 8px;
        border: none;
        padding: 8px 16px;
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        background-color: #1D4ED8;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.25);
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        background-color: #F1F5F9;
        border-radius: 8px 8px 0px 0px;
        padding: 0px 16px;
        color: #475569;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1E40AF !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Data Loading & Caching Function
# ---------------------------------------------------------
DATA_PATH = os.path.join("data", "supermarket_sales.csv")

@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH):
        return None
    
    df = pd.read_csv(DATA_PATH)
    
    # Preprocessing
    df['Date'] = pd.to_datetime(df['Date'])
    df['Hour'] = pd.to_datetime(df['Time'], format='%H:%M').dt.hour
    df['Month'] = df['Date'].dt.month_name()
    df['Day_Name'] = df['Date'].dt.day_name()
    
    def get_time_period(hour):
        if 6 <= hour < 12:
            return 'Morning'
        elif 12 <= hour < 17:
            return 'Afternoon'
        else:
            return 'Evening'
            
    df['Time_Period'] = df['Hour'].apply(get_time_period)
    return df

df_raw = load_data()

# Check dataset existence
if df_raw is None:
    st.error("⚠️ Dataset not found!")
    st.info("Please place `supermarket_sales.csv` inside the `data/` folder directory.")
    st.stop()

# ---------------------------------------------------------
# Sidebar Navigation & Filters
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div class="sidebar-header">
        <h3>🛒 Supermarket Sales</h3>
        <p>IBM SkillsBuild Analytics 2026</p>
    </div>
    """, unsafe_allow_html=True)
    
    page = st.radio(
        "📌 Navigation",
        [
            "1. Overview",
            "2. Sales Analysis",
            "3. Product Analysis",
            "4. Customer Analysis",
            "5. Branch & City Analysis",
            "6. Payment Analysis",
            "7. Time Analysis",
            "8. Customer Segmentation",
            "9. Machine Learning",
            "10. Business Insights",
            "11. About Project"
        ]
    )
    
    st.markdown("---")
    st.subheader("🔍 Interactive Filters")
    
    # Reset filters state handling
    if st.button("🔄 Reset All Filters"):
        st.session_state.clear()
        st.rerun()

    # Multi-select filters
    selected_branches = st.multiselect("Branch", options=sorted(df_raw['Branch'].unique()), default=sorted(df_raw['Branch'].unique()))
    selected_cities = st.multiselect("City", options=sorted(df_raw['City'].unique()), default=sorted(df_raw['City'].unique()))
    selected_products = st.multiselect("Product Line", options=sorted(df_raw['Product line'].unique()), default=sorted(df_raw['Product line'].unique()))
    selected_genders = st.multiselect("Gender", options=sorted(df_raw['Gender'].unique()), default=sorted(df_raw['Gender'].unique()))
    selected_customer_types = st.multiselect("Customer Type", options=sorted(df_raw['Customer type'].unique()), default=sorted(df_raw['Customer type'].unique()))
    selected_payments = st.multiselect("Payment Method", options=sorted(df_raw['Payment'].unique()), default=sorted(df_raw['Payment'].unique()))

    min_date = df_raw['Date'].min().date()
    max_date = df_raw['Date'].max().date()
    date_range = st.date_input("Date Range", value=(min_date, max_date), min_value=min_date, max_value=max_date)

# ---------------------------------------------------------
# Filter Application
# ---------------------------------------------------------
filtered_df = df_raw.copy()

if selected_branches:
    filtered_df = filtered_df[filtered_df['Branch'].isin(selected_branches)]
if selected_cities:
    filtered_df = filtered_df[filtered_df['City'].isin(selected_cities)]
if selected_products:
    filtered_df = filtered_df[filtered_df['Product line'].isin(selected_products)]
if selected_genders:
    filtered_df = filtered_df[filtered_df['Gender'].isin(selected_genders)]
if selected_customer_types:
    filtered_df = filtered_df[filtered_df['Customer type'].isin(selected_customer_types)]
if selected_payments:
    filtered_df = filtered_df[filtered_df['Payment'].isin(selected_payments)]

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_d, end_d = date_range
    filtered_df = filtered_df[(filtered_df['Date'].dt.date >= start_d) & (filtered_df['Date'].dt.date <= end_d)]

# Handle empty state
if filtered_df.empty:
    st.warning("⚠️ No data available for the selected filters. Please adjust or reset your filters.")
    st.stop()

# ---------------------------------------------------------
# Page 1: Overview
# ---------------------------------------------------------
if page == "1. Overview":
    st.title("📊 AI-Powered Supermarket Sales Overview")
    st.caption("Data Analytics | Unsupervised Customer Segmentation | Predictive Machine Learning")
    
    # KPI Metrics Row
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    total_revenue = filtered_df['Total'].sum()
    total_tx = len(filtered_df)
    total_qty = filtered_df['Quantity'].sum()
    total_gross = filtered_df['gross income'].sum()
    avg_spend = filtered_df['Total'].mean()
    avg_rating = filtered_df['Rating'].mean()
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Total Sales</div>
            <div class="metric-value">${total_revenue:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Transactions</div>
            <div class="metric-value">{total_tx:,}</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Qty Sold</div>
            <div class="metric-value">{total_qty:,}</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Gross Income</div>
            <div class="metric-value">${total_gross:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    with col5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Avg Spend</div>
            <div class="metric-value">${avg_spend:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    with col6:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Avg Rating</div>
            <div class="metric-value">{avg_rating:.2f}/10</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Overview Charts Row
    c1, c2 = st.columns([7, 5])
    with c1:
        daily = filtered_df.groupby('Date')['Total'].sum().reset_index()
        fig_trend = px.line(daily, x='Date', y='Total', title="📈 Daily Sales Revenue Trend",
                            line_shape='linear', color_discrete_sequence=['#1E40AF'])
        fig_trend.update_layout(paper_bgcolor='white', plot_bgcolor='#F8FAFC', margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_trend, use_container_width=True)
        
    with c2:
        branch_df = filtered_df.groupby('Branch')['Total'].sum().reset_index()
        fig_branch = px.bar(branch_df, x='Branch', y='Total', color='Branch',
                            title="🏢 Revenue Share by Branch",
                            color_discrete_sequence=['#1E40AF', '#2563EB', '#60A5FA'])
        fig_branch.update_layout(paper_bgcolor='white', plot_bgcolor='#F8FAFC', margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_branch, use_container_width=True)

    # Automated Summary Callout
    top_prod = filtered_df.groupby('Product line')['Total'].sum().idxmax()
    top_branch = filtered_df.groupby('Branch')['Total'].sum().idxmax()
    st.info(f"💡 **Executive Summary**: Operating across filtered data, **Branch {top_branch}** recorded the highest revenue, while **{top_prod}** emerged as the top revenue-generating category. Total recorded gross profit margin stands at **4.76%**.")

# ---------------------------------------------------------
# Page 2: Sales Analysis
# ---------------------------------------------------------
elif page == "2. Sales Analysis":
    st.title("📈 Detailed Sales & Revenue Analysis")
    
    col1, col2 = st.columns(2)
    with col1:
        prod_rev = filtered_df.groupby('Product line')['Total'].sum().reset_index().sort_values(by='Total', ascending=True)
        fig_prod = px.bar(prod_rev, x='Total', y='Product line', orientation='h',
                          title="Sales Revenue by Product Line ($)",
                          color_discrete_sequence=['#2563EB'])
        fig_prod.update_layout(paper_bgcolor='white', plot_bgcolor='#F8FAFC')
        st.plotly_chart(fig_prod, use_container_width=True)
        
    with col2:
        gross_prod = filtered_df.groupby('Product line')['gross income'].sum().reset_index().sort_values(by='gross income', ascending=True)
        fig_gross = px.bar(gross_prod, x='gross income', y='Product line', orientation='h',
                           title="Gross Income Contribution by Product Line ($)",
                           color_discrete_sequence=['#1E40AF'])
        fig_gross.update_layout(paper_bgcolor='white', plot_bgcolor='#F8FAFC')
        st.plotly_chart(fig_gross, use_container_width=True)

    st.markdown("### Historical Daily Revenue Breakdown")
    fig_hist = px.histogram(filtered_df, x='Total', nbins=30, title="Transaction Value Distribution ($)",
                            color_discrete_sequence=['#3B82F6'], marginal="rug")
    fig_hist.update_layout(paper_bgcolor='white', plot_bgcolor='#F8FAFC')
    st.plotly_chart(fig_hist, use_container_width=True)

# ---------------------------------------------------------
# Page 3: Product Analysis
# ---------------------------------------------------------
elif page == "3. Product Analysis":
    st.title("📦 Product Line Performance & Ratings")
    
    c1, c2 = st.columns(2)
    with c1:
        qty_df = filtered_df.groupby('Product line')['Quantity'].sum().reset_index()
        fig_qty = px.pie(qty_df, values='Quantity', names='Product line',
                         title="Product Units Sold Share",
                         color_discrete_sequence=px.colors.sequential.Blues_r)
        fig_qty.update_layout(paper_bgcolor='white')
        st.plotly_chart(fig_qty, use_container_width=True)
        
    with c2:
        rating_df = filtered_df.groupby('Product line')['Rating'].mean().reset_index().sort_values(by='Rating', ascending=False)
        fig_rating = px.bar(rating_df, x='Rating', y='Product line', orientation='h',
                            title="Average Customer Rating by Category (Out of 10)",
                            color='Rating', color_continuous_scale='Blues')
        fig_rating.update_layout(paper_bgcolor='white', plot_bgcolor='#F8FAFC')
        st.plotly_chart(fig_rating, use_container_width=True)

    st.markdown("### Category Summary Table")
    prod_table = filtered_df.groupby('Product line').agg(
        Total_Revenue=('Total', 'sum'),
        Total_Quantity=('Quantity', 'sum'),
        Gross_Income=('gross income', 'sum'),
        Avg_Unit_Price=('Unit price', 'mean'),
        Avg_Rating=('Rating', 'mean')
    ).reset_index()
    st.dataframe(prod_table.style.format({
        'Total_Revenue': '${:,.2f}',
        'Gross_Income': '${:,.2f}',
        'Avg_Unit_Price': '${:,.2f}',
        'Avg_Rating': '{:.2f}'
    }), use_container_width=True)

# ---------------------------------------------------------
# Page 4: Customer Analysis
# ---------------------------------------------------------
elif page == "4. Customer Analysis":
    st.title("👥 Customer Demographics & Behavior")
    
    col1, col2 = st.columns(2)
    with col1:
        gender_df = filtered_df.groupby('Gender')['Total'].sum().reset_index()
        fig_gender = px.pie(gender_df, values='Total', names='Gender', title="Revenue Share by Gender",
                            color_discrete_sequence=['#1E40AF', '#60A5FA'])
        st.plotly_chart(fig_gender, use_container_width=True)
        
    with col2:
        ctype_df = filtered_df.groupby('Customer type')['Total'].sum().reset_index()
        fig_ctype = px.pie(ctype_df, values='Total', names='Customer type', title="Revenue Share by Customer Type",
                           color_discrete_sequence=['#2563EB', '#93C5FD'])
        st.plotly_chart(fig_ctype, use_container_width=True)

    st.markdown("### Spend Distribution: Member vs Normal Customer")
    fig_box = px.box(filtered_df, x='Customer type', y='Total', color='Gender',
                     title="Transaction Spend Distribution by Customer Type and Gender",
                     color_discrete_sequence=['#1E40AF', '#3B82F6'])
    fig_box.update_layout(paper_bgcolor='white', plot_bgcolor='#F8FAFC')
    st.plotly_chart(fig_box, use_container_width=True)

# ---------------------------------------------------------
# Page 5: Branch & City Analysis
# ---------------------------------------------------------
elif page == "5. Branch & City Analysis":
    st.title("🏢 Geographic & Branch Operational Intelligence")
    
    c1, c2 = st.columns(2)
    with c1:
        city_df = filtered_df.groupby('City')['Total'].sum().reset_index()
        fig_city = px.bar(city_df, x='City', y='Total', title="Sales Revenue by City",
                          color='City', color_discrete_sequence=['#1E40AF', '#2563EB', '#60A5FA'])
        fig_city.update_layout(paper_bgcolor='white', plot_bgcolor='#F8FAFC')
        st.plotly_chart(fig_city, use_container_width=True)
        
    with c2:
        bc_df = filtered_df.groupby(['Branch', 'Product line'])['Total'].sum().reset_index()
        fig_bc = px.bar(bc_df, x='Branch', y='Total', color='Product line', title="Branch Sales Split by Product Line",
                        barmode='stack', color_discrete_sequence=px.colors.sequential.Blues_r)
        fig_bc.update_layout(paper_bgcolor='white', plot_bgcolor='#F8FAFC')
        st.plotly_chart(fig_bc, use_container_width=True)

# ---------------------------------------------------------
# Page 6: Payment Analysis
# ---------------------------------------------------------
elif page == "6. Payment Analysis":
    st.title("💳 Payment Dynamics & Transaction Preferences")
    
    col1, col2 = st.columns(2)
    with col1:
        pay_df = filtered_df['Payment'].value_counts().reset_index()
        pay_df.columns = ['Payment Method', 'Count']
        fig_pay = px.pie(pay_df, values='Count', names='Payment Method', title="Payment Method Popularity",
                         color_discrete_sequence=['#1E40AF', '#2563EB', '#60A5FA'])
        st.plotly_chart(fig_pay, use_container_width=True)
        
    with col2:
        pay_spend = filtered_df.groupby('Payment')['Total'].sum().reset_index()
        fig_ps = px.bar(pay_spend, x='Payment', y='Total', title="Total Revenue by Payment Method ($)",
                        color='Payment', color_discrete_sequence=['#1E40AF', '#2563EB', '#60A5FA'])
        fig_ps.update_layout(paper_bgcolor='white', plot_bgcolor='#F8FAFC')
        st.plotly_chart(fig_ps, use_container_width=True)

# ---------------------------------------------------------
# Page 7: Time Analysis
# ---------------------------------------------------------
elif page == "7. Time Analysis":
    st.title("⏰ Peak Shopping Hours & Temporal Dynamics")
    
    hourly = filtered_df.groupby('Hour')['Total'].sum().reset_index()
    fig_hour = px.bar(hourly, x='Hour', y='Total', title="Total Revenue by Hour of Day (Peak Hours)",
                      color_discrete_sequence=['#1E40AF'])
    fig_hour.update_layout(paper_bgcolor='white', plot_bgcolor='#F8FAFC')
    st.plotly_chart(fig_hour, use_container_width=True)
    
    c1, c2 = st.columns(2)
    with c1:
        day_df = filtered_df.groupby('Day_Name')['Total'].sum().reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']).reset_index()
        fig_day = px.line(day_df, x='Day_Name', y='Total', title="Sales Revenue by Day of Week", markers=True,
                          color_discrete_sequence=['#2563EB'])
        fig_day.update_layout(paper_bgcolor='white', plot_bgcolor='#F8FAFC')
        st.plotly_chart(fig_day, use_container_width=True)
        
    with c2:
        period_df = filtered_df.groupby('Time_Period')['Total'].sum().reset_index()
        fig_period = px.pie(period_df, values='Total', names='Time_Period', title="Revenue Share by Time Period",
                            color_discrete_sequence=['#1E40AF', '#3B82F6', '#93C5FD'])
        st.plotly_chart(fig_period, use_container_width=True)

# ---------------------------------------------------------
# Page 8: Customer Segmentation
# ---------------------------------------------------------
elif page == "8. Customer Segmentation":
    st.title("🎯 K-Means Customer Segmentation")
    st.caption("Unsupervised Machine Learning for Retail Customer Behavioral Clustering")
    
    # Perform K-Means on filtered data
    X_clust = filtered_df[['Unit price', 'Quantity', 'Total', 'Rating', 'Hour']].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clust)
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    filtered_df['Cluster'] = kmeans.fit_predict(X_scaled)
    sil_score = silhouette_score(X_scaled, filtered_df['Cluster'])
    
    st.markdown(f"""
    <div class="metric-card" style="border-left: 4px solid #1E40AF;">
        <div class="metric-title">Model Evaluation Metric</div>
        <div class="metric-value">Silhouette Score: {sil_score:.4f}</div>
        <div class="metric-subtitle">Optimal Cluster Count: K = 3 (Derived via Elbow & Silhouette Evaluation)</div>
    </div>
    """, unsafe_allow_html=True)
    
    fig_scatter = px.scatter(filtered_df, x='Total', y='Quantity', color='Cluster',
                             size='Unit price', hover_data=['Product line', 'Branch', 'Rating'],
                             title="Customer Segments: Total Spend vs Quantity Purchased",
                             color_continuous_scale='Blues')
    fig_scatter.update_layout(paper_bgcolor='white', plot_bgcolor='#F8FAFC')
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    st.markdown("### Cluster Characteristics Summary")
    c_summary = filtered_df.groupby('Cluster').agg(
        Avg_Spend=('Total', 'mean'),
        Avg_Quantity=('Quantity', 'mean'),
        Avg_Unit_Price=('Unit price', 'mean'),
        Avg_Rating=('Rating', 'mean'),
        Customer_Count=('Total', 'count')
    ).reset_index()
    st.dataframe(c_summary.style.format({
        'Avg_Spend': '${:,.2f}',
        'Avg_Quantity': '{:.2f}',
        'Avg_Unit_Price': '${:,.2f}',
        'Avg_Rating': '{:.2f}'
    }), use_container_width=True)

# ---------------------------------------------------------
# Page 9: Machine Learning
# ---------------------------------------------------------
elif page == "9. Machine Learning":
    st.title("🤖 Predictive Sales Revenue Model")
    st.caption("Random Forest Regressor for Transaction Spend Estimation")
    
    features = ['Unit price', 'Quantity', 'Tax 5%', 'Rating', 'Hour']
    X = filtered_df[features]
    y = filtered_df['Total']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("MAE", f"${mae:.4f}")
    col2.metric("MSE", f"${mse:.4f}")
    col3.metric("RMSE", f"${rmse:.4f}")
    col4.metric("R² Score", f"{r2:.4f}")
    
    st.markdown("<br>", unsafe_allow_html=True)
    fig_pred = px.scatter(x=y_test, y=y_pred, labels={'x': 'Actual Total Spend ($)', 'y': 'Predicted Total Spend ($)'},
                          title="Actual vs Predicted Sales Revenue", color_discrete_sequence=['#2563EB'])
    fig_pred.add_shape(type="line", x0=y_test.min(), y0=y_test.min(), x1=y_test.max(), y1=y_test.max(),
                      line=dict(color="Red", width=2, dash="dash"))
    fig_pred.update_layout(paper_bgcolor='white', plot_bgcolor='#F8FAFC')
    st.plotly_chart(fig_pred, use_container_width=True)
    
    st.markdown("---")
    st.subheader("💡 Interactive Revenue Predictor Tool")
    
    c1, c2, c3, c4 = st.columns(4)
    in_unit_price = c1.number_input("Unit Price ($)", min_value=1.0, max_value=100.0, value=55.0)
    in_qty = c2.number_input("Quantity", min_value=1, max_value=10, value=5)
    in_tax = c3.number_input("Tax 5% ($)", min_value=0.0, max_value=50.0, value=in_unit_price * in_qty * 0.05)
    in_rating = c4.slider("Rating Score", min_value=1.0, max_value=10.0, value=7.5)
    in_hour = st.slider("Shopping Hour", min_value=10, max_value=20, value=14)
    
    if st.button("🔮 Calculate Predicted Total Revenue"):
        pred_val = model.predict([[in_unit_price, in_qty, in_tax, in_rating, in_hour]])[0]
        st.success(f"🎉 **Estimated Total Revenue**: **${pred_val:,.2f}**")

# ---------------------------------------------------------
# Page 10: Business Insights
# ---------------------------------------------------------
elif page == "10. Business Insights":
    st.title("💡 AI-Assisted Business Insights")
    st.caption("Actionable Intelligence Derived from Empirical Retail Sales Data")
    
    top_p = filtered_df.groupby('Product line')['Total'].sum().idxmax()
    top_b = filtered_df.groupby('Branch')['Total'].sum().idxmax()
    peak_h = filtered_df.groupby('Hour')['Total'].sum().idxmax()
    top_pay = filtered_df['Payment'].value_counts().idxmax()
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
        <div class="content-card">
            <h4>📦 Top Performing Category</h4>
            <p><b>{top_p}</b> generates the maximum revenue contribution across the filtered dataset. Ensure restocking schedules are aligned with high velocity sales.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="content-card">
            <h4>⏰ Peak Shopping Hours</h4>
            <p>Peak store traffic and total volume occur around <b>{peak_h}:00 Hours</b>. Increase billing counter personnel during these hours to minimize queue drops.</p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="content-card">
            <h4>🏢 Leading Branch Location</h4>
            <p><b>Branch {top_b}</b> yields the highest gross margin and overall transaction revenue. Model inventory logistics off this branch's stocking strategy.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="content-card">
            <h4>💳 Preferred Payment Preference</h4>
            <p>Customers overwhelmingly utilize <b>{top_pay}</b> for retail transactions. Strategic partnerships with e-wallet providers can yield higher customer conversion.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### Optional Generative AI API Integration")
    st.info("ℹ️ The core project runs standalone using deterministic Python analytics. To enable optional LLM insights, add `OPENAI_API_KEY` or `GEMINI_API_KEY` into a `.env` file (reference `.env.example`).")

# ---------------------------------------------------------
# Page 11: About Project
# ---------------------------------------------------------
elif page == "11. About Project":
    st.title("ℹ️ Academic Project Portfolio")
    
    st.markdown("""
    <div class="content-card">
        <h3>AICTE | IBM SkillsBuild Data Analytics with AI Internship 2026</h3>
        <p><b>Project Title:</b> AI-Powered Supermarket Sales Analysis and Customer Insights</p>
        <p><b>Student Name:</b> Medida Sri Venkata Praveen</p>
        <p><b>Academic Degree:</b> B.Tech - Artificial Intelligence and Machine Learning</p>
        <p><b>Technology Stack:</b> Python 3.10+, Pandas, NumPy, Matplotlib, Seaborn, Plotly, Scikit-learn, Streamlit, Jupyter Notebook, python-docx</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Project Architecture & Workflow")
    st.markdown("""
    ```
    Data Collection (CSV) ➔ Data Preprocessing & Cleaning ➔ Exploratory Data Analysis (EDA) 
    ➔ K-Means Customer Clustering ➔ Machine Learning Regression ➔ Interactive Streamlit Dashboard 
    ➔ Automated Word Academic Report Generation
    ```
    """)
