import os
import glob
import pandas as pd

dataset_files="data/text/"

files=glob.glob(os.path.join(dataset_files, "*.csv.zip"))


print("Looking in:", os.path.abspath(dataset_files))
print("CSV files found:", files)

df=(pd.read_csv(file, compression='zip') for file in files )
combined=pd.concat(df,ignore_index=True )

full_dataset = combined.sample(frac=1, random_state=42).reset_index(drop=True)

output_file = os.path.join(dataset_files, "combined_dataset.csv")
full_dataset.to_csv(output_file, index=False)

print(f"Combined dataset saved at: {output_file}")
print("Shape of combined dataset:", full_dataset.shape)
