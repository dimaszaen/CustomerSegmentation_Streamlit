import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.cluster import KMeans
import pickle
import streamlit as st
import os
from datetime import datetime
import squarify
import base64

# GUI setup
st.title("OPTIMALISASI PENENTUAN PUSAT TITIK CLUSTERING K-MEANS DAN MODIFIKASI RFM UNTUK SEGMENTASI PELANGGAN RETAIL PADA PERUSAHAAN JASA PENGIRIMAN EXPRESS DALAM MENUNJANG STRATEGI PROMOSI")
st.header(" Tesis 2011600026 Dimas Zaen Fikri Amar", divider='rainbow')

menu = ["Business Understanding", "Data Understanding","Data preparation","Modeling & Evaluation","Predict"] # , "BigData: Spark"
choice = st.sidebar.selectbox('Menu', menu)

def load_data(uploaded_file):
    if uploaded_file is not None:
        st.sidebar.success("File uploaded successfully!")
        df = pd.read_csv(uploaded_file, encoding='latin-1', sep='\s+', header=None, 
                         names=['Cust_Branch', 'Cust_Phone','Cust_Name'	
                                'ReceiptNo','ReceiptDate',	
                                'Total_Package', 'Total_Weight',	
                                'Total_Received_Amount','Deleted_Flag'])
        df.to_csv("CDNOW_master_new.txt", index=False)
        df['ReceiptDate'] = pd.to_datetime(df['ReceiptDate'], format='%Y%m%d')
        st.session_state['df'] = df
        return df
    else:
        st.write("Please upload a data file to proceed.")
        return None

# Function to generate CSV download link
def csv_download_link(df, csv_file_name, download_link_text):
    csv_data = df.to_csv(index=True)
    b64 = base64.b64encode(csv_data.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{csv_file_name}">{download_link_text}</a>'
    st.markdown(href, unsafe_allow_html=True)    
# Initializing session state variables
if 'df' not in st.session_state:
    st.session_state['df'] = None

if 'uploaded_file' not in st.session_state:
    st.session_state['uploaded_file'] = None

# Main Menu
if choice == 'Business Understanding':
    st.subheader("Business Understanding")
    st.write("""
    Penelitian ini dimulai dengan kegiatan wawancara dan pengamatan langsung kepada C2C Manager dengan tujuan mengidentifikasi masalah yang ada. Studi pustaka kemudian dilakukan untuk mencari solusi dan metode dari penelitian terdahulu. Hasilnya adalah terdefinisinya rumusan masalah dan tujuan penelitian. Analisis masalah dilakukan dengan menggunakan tabel gap analysis, di mana harapan dan realita dibandingkan. Masalah yang diidentifikasi melibatkan penurunan jumlah transaksi pelanggan dan keinginan perusahaan untuk menerapkan strategi CRM, khususnya segmentasi pelanggan retail secara optimal. Tabel gap analysis menunjukkan bahwa strategi promosi yang dilakukan belum memperhatikan karakteristik pelanggan secara optimal.
    
    Tujuan penelitian adalah menghasilkan jumlah cluster optimal, mengoptimalkan proses inisialisasi pusat cluster, dan menganalisis cluster yang terbentuk berdasarkan karakteristik masing-masing. Pertanyaan penelitian mencakup jumlah optimal klaster, peningkatan performa K-Means melalui optimalisasi titik awal pusat klaster, dan karakteristik klaster yang terbentuk.

    """)
    #st.image("Customer-Segmentation.png", caption="Customer Segmentation", use_column_width=True)

    
elif choice == 'Data Understanding':    

   # Daftar semua file di folder 'sample_data'
    sample_files = os.listdir('data')
    
    # Create a radio button to allow users to choose between using a sample file or uploading a new file
    data_source = st.sidebar.radio('Data source', ['Use a sample file', 'Upload a new file'])
    
    if data_source == 'Use a sample file':
        # Allows the user to select a file from the list
        selected_file = st.sidebar.selectbox('Choose a sample file', sample_files)
        
       # Read the selected file (you will need additional logic to read the file here)
        file_path = os.path.join('data', selected_file)
        st.session_state['uploaded_file'] = open(file_path, 'r')
        load_data(st.session_state['uploaded_file'])

    else:
       # Allows users to upload a new file
        st.session_state['uploaded_file'] = st.sidebar.file_uploader("Choose a file", type=['txt'])
        
        if st.session_state['uploaded_file'] is not None:
            load_data(st.session_state['uploaded_file'])

    # st.session_state['uploaded_file'] = st.sidebar.file_uploader("Choose a file", type=['txt'])
    # load_data(st.session_state['uploaded_file'])
    
    if st.session_state['df'] is not None:
        st.write("### Data Overview")
        st.write("Number of rows:", st.session_state['df'].shape[0])
        st.write("Number of columns:", st.session_state['df'].shape[1])
        st.write("First five rows of the data:")
        st.write(st.session_state['df'].head())

elif choice == 'Data preparation': 
    st.write("### Data Cleaning")
    
    if st.session_state['df'] is not None:
        # 1. Handling missing, null, and duplicate values
        st.write("Number of missing values:")
        st.write(st.session_state['df'].isnull().sum())

        st.write("Number of NA values:")
        st.write((st.session_state['df'] == 'NA').sum())

        st.write("Number of duplicate rows:", st.session_state['df'].duplicated().sum())

        # Providing options for handling missing and duplicate values
        if st.checkbox('Remove duplicate rows'):
            st.session_state['df'].drop_duplicates(inplace=True)
            st.write("Duplicate rows removed.")
        
        if st.checkbox('Remove rows with NA values'):
            st.session_state['df'].replace('NA', pd.NA, inplace=True)
            st.session_state['df'].dropna(inplace=True)
            st.write("Rows with NA values removed.")
            
        # 2. Display number of unique values for each column
        st.write("Number of unique values for each column:")
        st.write(st.session_state['df'].nunique())

        # 3. Plotting distribution for numeric columns
        st.write("### Distribution plots")
        for col in st.session_state['df'].select_dtypes(include=['number']).columns:
            st.write(f"#### {col}")
            fig, ax = plt.subplots()
            st.session_state['df'][col].hist(ax=ax)
            st.pyplot(fig)

        # 4. Display boxplots for numeric columns
        st.write("### Boxplots for numeric columns")
        for col in st.session_state['df'].select_dtypes(include=['number']).columns:
            st.write(f"#### {col}")
            fig, ax = plt.subplots()
            st.session_state['df'].boxplot(column=col, ax=ax)
            st.pyplot(fig)

        # Additional Data Overview
        st.write("Transactions timeframe from {} to {}".format(st.session_state['df']['day'].min(), st.session_state['df']['day'].max()))
        st.write("{:,} transactions don't have a customer id".format(st.session_state['df'][st.session_state['df'].Customer_id.isnull()].shape[0]))
        st.write("{:,} unique customer_id".format(len(st.session_state['df'].Customer_id.unique())))

        # Add Data Transformation ['Customer_id', 'day', 'Quantity', 'Sales']
        st.write("### Data Transformation")
        # Group the data by Customer_id and sum the other columns, excluding 'day'
        user_grouped = st.session_state['df'].groupby('Customer_id').agg({'Quantity': 'sum', 'Sales': 'sum'})
        st.write("### User Grouped Data")
        st.write(user_grouped.head())

        # Create a new column for the month
        st.session_state['df']['month'] = st.session_state['df']['day'].values.astype('datetime64[M]')
        st.write("### Data with Month Column")
        st.write(st.session_state['df'].head())

        # Plot the total Sales per month
        st.write("### Total Sales per Month")
        dfm = st.session_state['df'].groupby('month')['Quantity'].sum()
        st.line_chart(dfm)

        # Plot the total Quantity per month
        st.write("### Total Quantity per Month")
        dfpc = st.session_state['df'].groupby('month')['Sales'].sum()
        st.line_chart(dfpc)

        # ... (rest of your code, don't forget to modify scatter plots too)

        st.write("### Scatter Plot: Sales vs Quantity for Individual Transactions")
        fig, ax = plt.subplots()
        ax.scatter(st.session_state['df']['Sales'], st.session_state['df']['Quantity'])
        st.pyplot(fig)

        st.write("### Scatter Plot: Sales vs Quantity for User Grouped Data")
        fig, ax = plt.subplots()
        ax.scatter(user_grouped['Sales'], user_grouped['Quantity'])
        st.pyplot(fig)
    else:
        st.write("No data available. Please upload a file in the 'Data Understanding' section.")
    
   

elif choice == 'Modeling & Evaluation':
    st.write("### Modeling With KMeans")
    if st.session_state['df'] is not None:
        # RFM Analysis
        recent_date = st.session_state['df']['day'].max()

        # Calculate Recency, Frequency, and Monetary value for each customer
        df_RFM = st.session_state['df'].groupby('Customer_id').agg({
            'day': lambda x: (recent_date - x.max()).days, # Recency
            'Customer_id': 'count', # Frequency
            'Sales': 'sum' # Monetary
        }).rename(columns={'day': 'Recency', 'Customer_id': 'Frequency', 'Sales': 'Monetary'})

        st.title('KMeans Analysis using Elbow Method')

        # Build and display Elbow Method charts
        sse = {}
        for k in range(1, 20):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(df_RFM)
            sse[k] = kmeans.inertia_

        fig, ax = plt.subplots()
        ax.set_title('Elbow method')
        ax.set_xlabel('Number of clusters (k)')
        ax.set_ylabel('Sum of Squares of distances')
        sns.pointplot(x=list(sse.keys()), y=list(sse.values()), ax=ax)
        st.pyplot(fig)

        # Allows the user to choose the number of clusters k
        n_clusters = st.sidebar.number_input('Choose the number of clusters k from 2 to 20:', min_value=2, max_value=20, value=3, step=1, key="cluster_value")
        st.write(f'You have selected division {n_clusters} cluster.')

        # Apply the KMeans model to the selected number of clusters
        model = KMeans(n_clusters=n_clusters, random_state=42)
        model.fit(df_RFM)

        df_sub = df_RFM.copy()
        df_sub['Cluster'] = model.labels_

        #Descriptive statistics and statistics by cluster
        cluster_stats = df_sub.groupby('Cluster').agg({
            'Recency': 'mean',
            'Frequency': 'mean',
            'Monetary': ['mean', 'count']
        }).round(2)

        cluster_stats.columns = ['RecencyMean', 'FrequencyMean', 'MonetaryMean', 'Count']
        cluster_stats['Percent'] = (cluster_stats['Count'] / cluster_stats['Count'].sum() * 100).round(2)

        # Reset index so 'Cluster' becomes a regular column, instead of index
        cluster_stats.reset_index(inplace=True)

       # Rename cluster groups for better readability
        cluster_stats['Cluster'] = 'Cluster ' + cluster_stats['Cluster'].astype('str')

        st.subheader('Statistics by each Cluster')
        st.dataframe(cluster_stats)

        # Biểu đồ Scatter
        fig_scatter = px.scatter(
            cluster_stats,
            x='RecencyMean',
            y='MonetaryMean',
            size='FrequencyMean',
            color='Cluster',
            log_x=True,
            size_max=60
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        #Tree Map chart
        # Set color for each cluster - you can change this at your discretion
        colors_dict = {
            0: 'green',
            1: 'red',
            2: 'royalblue',
            3: 'orange',
            4: 'purple'
        }
        fig_treemap, ax_treemap = plt.subplots()  # Create separate fig and ax objects for the Tree Map chart
        fig_treemap.set_size_inches(14, 10)

        squarify.plot(sizes=cluster_stats['Count'], 
                    label=[f'Cụm {i}\n{row.RecencyMean} ngày\n{row.FrequencyMean} đơn hàng\n{row.MonetaryMean} $\n{row.Count} khách hàng ({row.Percent}%)' 
                            for i, row in cluster_stats.iterrows()],
                    color=[colors_dict.get(cluster) for cluster in cluster_stats.index],
                    alpha=0.6,
                    text_kwargs={'fontsize':12, 'fontweight':'bold'})

        ax_treemap.set_title("Phân Khúc Khách Hàng", fontsize=26, fontweight="bold")
        ax_treemap.axis('off')
        st.pyplot(fig_treemap)

        # Draw a 3D scatter plot chart
        fig_3d = px.scatter_3d(
            cluster_stats,
            x='RecencyMean',
            y='FrequencyMean',
            z='MonetaryMean',
            color='Cluster',
            size='Count',
            labels={'RecencyMean': 'Recency', 'FrequencyMean': 'Frequency', 'MonetaryMean': 'Monetary'}
        )

        st.plotly_chart(fig_3d, use_container_width=True)

        # Add button to export model
        if st.button('Export Model'):
            # Save the model to a .pkl file
            with open('kmeans_model.pkl', 'wb') as f:
                pickle.dump((model, cluster_stats), f)
            
            st.session_state.model_exported = True
            st.write('The model (kmeans_model.pkl) was exported successfully!')

        
    else:
        st.write("No data available. Please upload a file in the 'Data Understanding' section.")

elif choice == 'Predict':
    
    if 'model_exported' in st.session_state and st.session_state.model_exported:
       # Reload model and cluster_stats
        with open('kmeans_model.pkl', 'rb') as f:
            model, cluster_stats = pickle.load(f)

        st.subheader('Statistics by each Cluster')
        st.dataframe(cluster_stats)
        
        # New section added to receive data from users and make predictions
        st.subheader("Cluster Prediction for a New Customer")
                
        # Receive data from user
        customer_name = st.text_input('Customer Name:')
        recent_date = st.date_input('Date of most recent purchase:')
        quantity = st.number_input('Quantity:', min_value=0)
        monetary = st.number_input('Monetary:', min_value=0.0)
        
        if 'df_new' not in st.session_state:
            st.session_state['df_new'] = pd.DataFrame(columns=['Customer_id', 'day', 'Quantity', 'Sales'])

        if st.button("Add"):
            new_data = pd.DataFrame({'Customer_id': [customer_name], 'day': [recent_date], 'Quantity': [quantity], 'Sales': [monetary]})
            if 'df_new' not in st.session_state:
                st.session_state['df_new'] = new_data
            else:
                st.session_state['df_new'] = pd.concat([st.session_state['df_new'], new_data], ignore_index=True)
            
        st.write("Added Data")
        st.dataframe(st.session_state['df_new'])  # Hiển thị DataFrame sau khi người dùng nhấn "Add"

        # When the user presses the "Predict" button, cluster prediction is performed
        if st.button("Forecast"):
            # Calculate Recency, Frequency, and Monetary values
            recent_date = pd.Timestamp.now().date()  # Update current date
            df_RFM = st.session_state['df_new'].groupby('Customer_id').agg({
                'day': lambda x: (recent_date - x.max()).days,  # Recency
                'Customer_id': 'count',  # Frequency
                'Sales': 'sum'  # Monetary
            }).rename(columns={'day': 'Recency', 'Customer_id': 'Frequency', 'Sales': 'Monetary'})

            # Predict clusters using the trained model
            cluster_pred = model.predict(df_RFM)
            
           # Add prediction column to df_RFM
            df_RFM['Cluster'] = cluster_pred

            # Display the resulting DataFrame
            st.write("Predicted results:")
            st.dataframe(df_RFM)
            
            # Allow users to download results as CSV
            csv_download_link(df_RFM, 'RFM_prediction_results.csv', 'Download prediction results')
        
    else:
        st.write("You must export the model before making predictions.")
