import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load your CSV file
csv_file = "YOUR CSV FILE PATH GOES HERE"
data = pd.read_csv(csv_file)

# Print a summary of the dataset
print(data.describe())

# Print all of the Column Names in the CSV file
print(data.columns)

# Dimensionality Reduction using PCA
pca = PCA(n_components=2)
scaler = StandardScaler()

# Select only the numeric columns
numeric_cols = data.select_dtypes(include=np.number).columns.tolist()

scaled_data = scaler.fit_transform(data[numeric_cols].dropna())  # Drop rows with missing values
principal_components = pca.fit_transform(scaled_data)
principal_df = pd.DataFrame(data=principal_components, columns=["PC1", "PC2"])

# Visualize the PCA results
fig = px.scatter(principal_df, x="PC1", y="PC2")
fig.update_layout(title="PCA Scatterplot")
fig.show()

# Parallel Coordinates Plot
fig = px.parallel_coordinates(data, color=data.columns[0])
fig.update_layout(title="Parallel Coordinates Plot")
fig.show()

# 3D Scatterplot
x_column = 'your_x_column'
y_column = 'your_y_column'
z_column = 'your_z_column'
fig = px.scatter_3d(data, x=x_column, y=y_column, z=z_column, color='your_color_column')
fig.update_layout(title=f'3D Scatterplot: {x_column} vs {y_column} vs {z_column}')
fig.show()

# Sunburst Chart
fig = px.sunburst(data, path=['your_level1_column', 'your_level2_column', 'your_level3_column'], values='your_value_column')
fig.update_layout(title="Sunburst Chart")
fig.show()

# Box Plot
fig = px.box(data, x="your_categorical_column", y="your_numeric_column")
fig.update_layout(title="Box Plot")
fig.show()

# Violin Plot
fig = px.violin(data, x="your_categorical_column", y="your_numeric_column", box=True, points="all")
fig.update_layout(title="Violin Plot")
fig.show()
