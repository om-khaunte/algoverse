import pandas as pd

df = pd.read_csv("data\\selected\\salmonelladataset.csv")

print(df[["Genome ID", "Genome Name"]].head(20).to_string(index=False))