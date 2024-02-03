from sys import displayhook
from tkinter.tix import DisplayStyle
from matplotlib import cm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.calibration import LabelEncoder
from sklearn.cluster import KMeans
import pickle
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import confusion_matrix, silhouette_samples, silhouette_score
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
import os
from datetime import datetime
import squarify
import base64
from scipy.stats import entropy


# GUI setup
st.title("OPTIMALISASI PENENTUAN PUSAT TITIK CLUSTERING K-MEANS DAN MODIFIKASI RFM UNTUK SEGMENTASI PELANGGAN RETAIL PADA PERUSAHAAN JASA PENGIRIMAN EXPRESS DALAM MENUNJANG STRATEGI PROMOSI")
st.header(" Tesis 2011600026 Dimas Zaen Fikri Amar", divider='rainbow')

menu = ["Business Understanding", "Data Understanding","Data preparation","Modeling & Evaluation"] # , "BigData: Spark"
choice = st.sidebar.selectbox('Menu', menu)

def load_data(uploaded_file):
    if uploaded_file is not None:
        st.sidebar.success("File uploaded successfully!")
        df = pd.read_csv(uploaded_file, encoding='latin-1', sep=';', header=None, 
                         names=['Cust_Branch', 'Cust_Phone','Cust_Name'	,
                                'ReceiptNo','ReceiptDate',	
                                'Total_Package', 'Total_Weight',	
                                'Total_Received_Amount','Deleted_Flag'])
        df.to_csv("CDNOW_master_new.txt", index=False)
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

    st.title("### Data Understanding")
   # Daftar semua file di folder 'sample_data'
    sample_files = os.listdir('data')
    
    # Create a radio button to allow users to choose between using a sample file or uploading a new file
    data_source = st.sidebar.radio('Data source', ['Upload a new file'])
    
    if data_source == 'Use a sample file':
        # Allows the user to select a file from the list
        selected_file = st.sidebar.selectbox('Choose a sample file', sample_files)
        
       # Read the selected file (you will need additional logic to read the file here)
        file_path = os.path.join('data', selected_file)
        st.session_state['uploaded_file'] = open(file_path, 'r')
        load_data(st.session_state['uploaded_file'])

    else:
       # Allows users to upload a new file
        st.session_state['uploaded_file'] = st.sidebar.file_uploader("Choose a file", type=['csv'])
        
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
    
    st.title("### Data Preparation") 
    st.write("### Data Cleaning")
    if st.session_state['df'] is not None:
        
        df= st.session_state['df']
        df = df[df['Deleted_Flag'] != 'Y']
        df = df[(~df['Cust_Name'].str.contains('ittest', case=False, na=False))]
        df = df[df['Total_Received_Amount'] > 0]
        df = df[df['Total_Package'] > 0]
        
        st.write(df.count())
        
        
        
    else:
        st.write("No data available. Please upload a file in the 'Data Understanding' section.")
    
    st.write("### Selection Data")
    st.session_state['df'][['Cust_Phone', 'Cust_Name', 'ReceiptNo', 'ReceiptDate', 'Total_Package', 'Total_Weight', 'Total_Received_Amount']]
  
    
    st.write("### Transformasi Data")
    # For recency will check what was the last date of transaction
    #First will convert the InvoiceDate as date variable

    df = pd.DataFrame(df)
    df['Receipt_Date'] = pd.to_datetime(df['ReceiptDate'])
    df['Receipt_Date'].max()

    # Tanggal analisis
    analysis_date =  pd.to_datetime('2024-01-01') #datetime.now()

    # Recency (R)
    df['Recency'] = (analysis_date - df.groupby('Cust_Phone')['Receipt_Date'].transform('max')).dt.days

    # Frequency (F)
    frequency_df = df.groupby('Cust_Phone')['Receipt_Date'].count().reset_index()
    frequency_df.columns = ['Cust_Phone', 'Frequency']
    df = pd.merge(df, frequency_df, on='Cust_Phone', how='left')

    # Monetary (M)
    monetary_df = df.groupby('Cust_Phone')['Total_Received_Amount'].sum().reset_index()
    monetary_df.columns = ['Cust_Phone', 'Monetary']
    df = pd.merge(df, monetary_df, on='Cust_Phone', how='left')

    # Menambahkan Weight
    quantity_df = df.groupby('Cust_Phone')[['Total_Package']].sum().reset_index()
    quantity_df.columns = ['Cust_Phone', 'Quantity']
    df = pd.merge(df, quantity_df, on='Cust_Phone', how='left')

    # Menambahkan Weight
    weight_df = df.groupby('Cust_Phone')[['Total_Weight']].sum().reset_index()
    weight_df.columns = ['Cust_Phone', 'Weight']
    df = pd.merge(df, weight_df, on='Cust_Phone', how='left')

    # Tampilkan hasil
    st.write(df[['Cust_Phone', 'Recency', 'Frequency', 'Monetary','Quantity', 'Weight']])
    
    st.write("### Normalisasi Data")
    # Calculate 'Length' based on the difference between the current date and the minimum date
    df['Receipt_Date'] = pd.to_datetime(df['ReceiptDate'])
    analysis_date = pd.to_datetime('2024-01-01')

    # Select the columns you want to normalize
    columns_to_normalize = df[['Recency', 'Frequency', 'Monetary', 'Quantity', 'Weight']]

    # Initialize the MinMaxScaler
    scaler = MinMaxScaler()

    # Fit and transform the selected columns
    df_normalized = pd.DataFrame(scaler.fit_transform(df[columns_to_normalize]), columns=columns_to_normalize)

    # Combine the normalized columns with the rest of the DataFrame
    df = pd.concat([df.drop(columns=columns_to_normalize), df_normalized], axis=1)
    
    st.session_state['df'] = df

    # Tampilkan hasil
    st.write(df[['Cust_Phone', 'Recency', 'Frequency', 'Monetary','Quantity', 'Weight']])
    
    
   
elif choice == 'Modeling & Evaluation':
    st.title("### Modeling & Evaluation") 

    if st.session_state['df'] is not None:
        
       # Statistik Deskriptif
        st.write("### Pemodelan RFM Modifikasi")
        
        st.write("Statistik Deskriptif")
        st.write(st.session_state['df'].describe())
        df= st.session_state['df']
        df = pd.DataFrame(df)
        
        st.write("RFM Modifikasi Normalisasi")
        st.write(df[['Recency', 'Frequency', 'Monetary','Quantity', 'Weight']])
        
        st.write("### Pemodelan Clustering K-Means ")
        
        
        
    if st.session_state['df'] is not None:
        
        
        st.write("### Mencari nilai-K Optimal")
        # Calculate Recency, Frequency, and Monetary value for each customer
        df_RFM = df[['Recency', 'Frequency', 'Monetary','Quantity', 'Weight']]
        
    
        # Allows the user to choose the number of clusters k
        n_clusters = st.sidebar.number_input('Choose the number of clusters k from 2 to 20:', min_value=2, max_value=20, value=3, step=1, key="cluster_value")
        st.write(f'You have selected division {n_clusters} cluster.')
        
        
        # Normalisasi data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_RFM)

        # Hitung WSS untuk berbagai jumlah klaster (misalnya, dari 1 hingga 10)
        wss_values = []
        for k in range(1, 11):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(X_scaled)
            wss_values.append(kmeans.inertia_)

        # Build and display Elbow Method charts
        sse = {}
        for k in range(1, 11):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(df_RFM)
            sse[k] = kmeans.inertia_

        fig, ax = plt.subplots()
        ax.set_title('Elbow method')
        ax.set_xlabel('Number of clusters (k)')
        ax.set_ylabel('Sum of Squares of distances')
        sns.pointplot(x=list(sse.keys()), y=list(sse.values()), ax=ax)
        sns.pointplot(x=list(range(1, 11)), y=wss_values, ax=ax)  # Use the range 1 to 10 for x-axis
        ax.axvline(x=n_clusters, color='r', linestyle='--', label=f'Chosen k={n_clusters}')
        st.pyplot(fig)
        
        # Menampilkan DataFrame dengan hasil klaster
        #clustered_df = df[['Recency', 'Frequency', 'Monetary', 'Quantity', 'Weight', 'Cluster']]
        st.write(df)
  
        st.write("### Pembuatan Alias Segmentasi")
        
        st.write(f'You have selected division {n_clusters} cluster.')

        
        # Perform K-Means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        df['Cluster'] = kmeans.fit_predict(df[['Recency', 'Frequency', 'Monetary','Quantity','Weight']])
        
        df = pd.DataFrame(df)
        st.write("Data Cluster")
        original_data = df[['Cluster','Recency', 'Frequency', 'Monetary','Quantity','Weight']]
        
        evaluated_df_grouped = original_data.groupby('Cluster').agg({
            'Recency': 'mean',      # Mean of Recency in each group
            'Frequency': 'mean',    # Mean of Frequency in each group
            'Monetary': 'mean',    # Mean of Frequency in each group
            'Quantity': 'mean',     # Mean of Quantity in each group
            'Weight': 'mean'    # Take the first Cluster value (assuming it's the same for all rows in the group)
         
        }).reset_index()
        
        st.write(evaluated_df_grouped)
        
        # Define customer group names
        group_names = {
            0: "Brown",
            1: "Silver",
            2: "Gold",
            3: "Diamond",
        }
        
        # Assign customer group names based on the cluster
        df['Customer Group'] = df['Cluster'].map(group_names)
        
        # Sort the DataFrame by the 'Cluster' column
        df.sort_values('Cluster', inplace=True)

        # Size of each group
        group_size = df['Customer Group'].value_counts().sort_index()
        # Average LRFMP variables per group
        average_lrfmp_per_group = df.groupby('Customer Group')[['Recency', 'Frequency', 'Monetary','Quantity','Weight']].mean()
        # Rename the columns
        average_lrfmp_per_group = average_lrfmp_per_group.rename(columns={
            'Recency': 'R_Avg',
            'Frequency': 'F_Avg',
            'Monetary': 'M_Avg',
            'Quantity': 'Q_Avg',
            'Weight': 'W_Avg'
        })
        
        # RFMQW scores with symbols for upper and lower bounds
        interpretation = lambda x: f"↑ {x + 0.1:.2f} ↓ {x - 0.1:.2f}"  # Example: Upper bound +0.1, Lower bound -0.1
        lrfmp_scores_with_symbols = average_lrfmp_per_group.applymap(interpretation)

        # Percentage of each group size
        group_size_percentage = (group_size / len(df)) * 100

        #st.write(pd.concat([group_size, group_size_percentage, average_lrfmp_per_group, lrfmp_scores_with_symbols], axis=1).fillna(0))
        st.write("Group Size")
        st.write(group_size)
        st.write("Average")
        st.write(average_lrfmp_per_group)  
        st.write("Score")     
        st.write(lrfmp_scores_with_symbols)
        st.write("Precentage")
        st.write(group_size_percentage)
        
        df = pd.DataFrame(df)
        st.write("Cluster With Alias")
        
        data_with_alias = df[['Customer Group','Cust_Phone','Cluster','Recency', 'Frequency', 'Monetary','Quantity','Weight']]

                # Assuming evaluated_df is your DataFrame
        evaluated_df_grouped = data_with_alias.groupby('Customer Group').agg({
            'Cluster': 'mean',
            'Cust_Phone': 'count',  # Count of rows in each group
            'Recency': 'mean',      # Mean of Recency in each group
            'Frequency': 'mean',    # Mean of Frequency in each group
            'Quantity': 'mean',     # Mean of Quantity in each group
            'Weight': 'mean'    # Take the first Cluster value (assuming it's the same for all rows in the group)
         
        }).reset_index()
        
        st.write(evaluated_df_grouped)
        
        
        # Menentukan fitur untuk visualisasi box plot customer group
        features_for_boxplot = ['Recency', 'Frequency', 'Monetary', 'Quantity', 'Weight']

        # Membuat figure dengan ukuran tertentu
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 8))

        # Menampilkan box plot untuk setiap fitur dalam setiap kelompok pelanggan
        for i, feature in enumerate(features_for_boxplot):
            row_index = i // 3
            col_index = i % 3
            ax = axes[row_index, col_index]
            
            sns.boxplot(x='Customer Group', y=feature, data=df, palette='viridis', ax=ax)
            ax.set_title(f'Box Plot of {feature} across Customer Groups')

        # Mengatur layout dan menampilkan plot
        plt.tight_layout()
        plt.show()
        st.pyplot(fig)
  
  
        st.write('## ##Evaluasi : Silhoutte,Purity dan Entropy')
        # Memilih fitur untuk klasterisasi (RFMQW)
        features = df[['Recency', 'Frequency', 'Monetary', 'Quantity', 'Weight']]

        st.write(f'You have selected division {n_clusters} cluster.')

   
        # Pelatihan model K-Means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        df['Cluster'] = kmeans.fit_predict(df[['Recency', 'Frequency', 'Monetary','Quantity','Weight']])
        

        # Evaluasi dengan metode silhouette
        silhouette_avg = silhouette_score(features, df['Cluster'])

        # Evaluasi dengan metode entropy
        label_encoder = LabelEncoder()
        true_labels_numeric = label_encoder.fit_transform(df['Customer Group'])
        predicted_labels = df['Cluster']
        cluster_distribution = np.bincount(predicted_labels) / len(predicted_labels)
        entropy_value = entropy(cluster_distribution, base=2)

        # Menggunakan confusion matrix untuk menghitung Purity
        conf_matrix = confusion_matrix(true_labels_numeric, predicted_labels)
        purity = np.sum(np.amax(conf_matrix, axis=0)) / np.sum(conf_matrix)

        # Menambahkan hasil evaluasi ke DataFrame
        df['Silhouette'] = silhouette_avg
        df['Entropy'] = entropy_value
        df['Purity'] = purity

        # # Menampilkan DataFrame dengan hasil evaluasi
        evaluated_df = df[['Cust_Phone','Customer Group', 'Recency', 'Frequency', 'Quantity', 'Weight', 'Cluster', 'Silhouette', 'Purity', 'Entropy']]
        # st.write(evaluated_df)
        
        # Assuming evaluated_df is your DataFrame
        evaluated_df_grouped = evaluated_df.groupby('Customer Group').agg({
            'Cust_Phone': 'count',  # Count of rows in each group
            'Recency': 'mean',      # Mean of Recency in each group
            'Frequency': 'mean',    # Mean of Frequency in each group
            'Quantity': 'mean',     # Mean of Quantity in each group
            'Weight': 'mean',       # Mean of Weight in each group
            'Cluster': 'first',     # Take the first Cluster value (assuming it's the same for all rows in the group)
            'Silhouette': 'mean',   # Mean of Silhouette in each group
            'Purity': 'mean',       # Mean of Purity in each group
            'Entropy': 'mean'       # Mean of Entropy in each group
        }).reset_index()

        st.write(evaluated_df_grouped)

        # Visualisasi Purity, Silhouette, dan Entropy
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Purity per Cluster
        cluster_labels = np.unique(predicted_labels)
        purity_values = [np.max(conf_matrix[:, cluster_label]) / np.sum(conf_matrix[:, cluster_label]) for cluster_label in cluster_labels]

        axes[0].bar(cluster_labels, purity_values, color='blue')
        axes[0].set_title('Purity per Cluster')
        axes[0].set_xlabel('Cluster')
        axes[0].set_ylabel('Purity')

        # Silhouette plot for each cluster
        sample_silhouette_values = silhouette_samples(features, df['Cluster'])
        y_lower = 10

        for i in range(k):
            ith_cluster_silhouette_values = sample_silhouette_values[df['Cluster'] == i]
            ith_cluster_silhouette_values.sort()
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
            color = cm.nipy_spectral(float(i) / k)
            axes[1].fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values,
                                facecolor=color, edgecolor=color, alpha=0.7)
            axes[1].text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            y_lower = y_upper + 10

        axes[1].set_title('Silhouette plot for each cluster')
        axes[1].set_xlabel('Silhouette score')
        axes[1].set_ylabel('Cluster')

        # Entropy per Cluster
        entropy_values = [entropy(conf_matrix[:, cluster_label], base=2) for cluster_label in cluster_labels]

        axes[2].bar(cluster_labels, entropy_values, color='blue')
        axes[2].set_title('Entropy per Cluster')
        axes[2].set_xlabel('Cluster')
        axes[2].set_ylabel('Entropy')

        plt.tight_layout()
        plt.show()
        st.pyplot(fig)
        

    else:
        st.write("No data available. Please upload a file in the 'Data Understanding' section.")


