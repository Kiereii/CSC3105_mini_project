import pandas as pd

# Check the structure of the first few rows of the dataset
df = pd.read_csv('/home/keyreii/Documents/Data Analytics Mini Project/Dataset/UWB-LOS-NLOS-Data-Set/dataset/uwb_dataset_part1.csv')

print("Dataset shape:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())
print("\nFirst 5 rows:")
print(df.head())
print("\nData types:")
print(df.dtypes)
print("\nBasic statistics:")
print(df.describe())