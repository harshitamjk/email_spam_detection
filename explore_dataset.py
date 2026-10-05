import pandas as pd

df = pd.read_csv("dataset/emails.csv")

print("Dataset shape:", df.shape)

print("\nMissing values:")
print(df.isnull().sum())

print("\nLabel distribution:")
print(df["label"].value_counts())

print("\nDuplicate rows:", df.duplicated().sum())

# Remove emails with no body
df = df.dropna(subset=["body"])

# Replace missing subjects with an empty string
df["subject"] = df["subject"].fillna("")

# Remove exact duplicate rows
df = df.drop_duplicates()

print("\nAfter cleaning:")
print("Dataset shape:", df.shape)
print("Missing values:")
print(df.isnull().sum())
print("Duplicates:", df.duplicated().sum())

print("\nCleaned label distribution:")
print(df["label"].value_counts())

#df.to_csv("dataset/emails_cleaned.csv", index=False)
#print("\nCleaned dataset saved successfully!")

df["text"] = df["subject"] + " " + df["body"]

print("\nText column created!")
print(df[["subject", "body", "text", "label"]].head(2))
df.to_csv("dataset/emails_cleaned.csv", index=False)

print("\nFinal cleaned dataset saved!")