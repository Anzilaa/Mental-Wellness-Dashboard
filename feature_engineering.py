import pandas as pd

# Load the dataset
input_path = 'archive/Mental_Health_Lifestyle_Dataset.csv'
df = pd.read_csv(input_path)

# Feature Engineering: Add Age Group
bins = [0, 17, 25, 35, 45, 60, 120]
labels = ['<18', '18-25', '26-35', '36-45', '46-60', '61+']
df['Age Group'] = pd.cut(df['Age'], bins=bins, labels=labels, right=True)

# Save the new dataset with engineered features
output_path = 'archive/Mental_Health_Lifestyle_Dataset_with_features.csv'
df.to_csv(output_path, index=False)

print('Feature engineering complete. New file saved as:', output_path)
