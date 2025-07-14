import pandas as pd

# Set the path to your Excel file
file_path = "MajorProject Datasets.xlsx"

# List of commodity sheets to include
commodity_sheets = [
    'Arecanut', 'Black Gram', 'Coconut', 'Dry Grapes', 'Green Gram',
    'Jaggery', 'Mustard', 'Onion', 'Potato', 'Ragi',
    'Rice', 'Soyabean', 'Wheat', 'Garlic', 'Groundnut'
]

# Read and merge all sheets
merged_data = []

for sheet in commodity_sheets:
    df = pd.read_excel(file_path, sheet_name=sheet)
    df['Commodity'] = sheet  # Add sheet name as a new column
    merged_data.append(df)

# Concatenate all into one DataFrame
combined_df = pd.concat(merged_data, ignore_index=True)

# Save as CSV
combined_df.to_csv("Merged_Commodities.csv", index=False)

print("✅ Merged all sheets into 'Merged_Commodities.csv'")
