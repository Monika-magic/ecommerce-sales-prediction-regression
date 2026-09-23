import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv("supermarket_sales_50000.csv")

print("\n===== DATASET INFORMATION =====")
print("Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

print("\nData types:")
print(df.dtypes)


# ============================================================
# 2. DATA QUALITY CHECK
# ============================================================

print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

print("\n===== DUPLICATES =====")
print("Number of duplicate rows:", df.duplicated().sum())


# ============================================================
# 3. STATISTICAL SUMMARY
# ============================================================

print("\n===== STATISTICAL SUMMARY =====")
print(df.describe())


# ============================================================
# 4. CATEGORICAL DATA ANALYSIS
# ============================================================

print("\n===== STORE DISTRIBUTION =====")
print(df["Store_ID"].value_counts())

print("\n===== PRODUCT CATEGORY DISTRIBUTION =====")
print(df["Product_Category"].value_counts())

print("\n===== PRODUCT DISTRIBUTION =====")
print(df["Product"].value_counts())

print("\n===== DAY DISTRIBUTION =====")
print(df["Day"].value_counts())


# ============================================================
# 5. CORRELATION WITH SALES
# ============================================================

print("\n===== CORRELATION WITH SALES =====")

numeric_columns = [
    "Quantity",
    "Unit_Price",
    "Discount",
    "Customer_Count",
    "Sales"
]

correlation = df[numeric_columns].corr()

print(correlation["Sales"].sort_values(ascending=False))


# ============================================================
# 6. EDA VISUALIZATIONS
# ============================================================

# Sales distribution
plt.figure(figsize=(8, 5))
plt.hist(df["Sales"], bins=30)
plt.title("Sales Distribution")
plt.xlabel("Sales")
plt.ylabel("Frequency")
plt.show()


# Sales boxplot
plt.figure(figsize=(8, 5))
plt.boxplot(df["Sales"])
plt.title("Sales Boxplot")
plt.ylabel("Sales")
plt.show()


# Average Sales by Product Category
category_sales = df.groupby("Product_Category")["Sales"].mean()

plt.figure(figsize=(8, 5))
category_sales.plot(kind="bar")
plt.title("Average Sales by Product Category")
plt.xlabel("Product Category")
plt.ylabel("Average Sales")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Average Sales by Day
day_sales = df.groupby("Day")["Sales"].mean()

plt.figure(figsize=(8, 5))
day_sales.plot(kind="bar")
plt.title("Average Sales by Day")
plt.xlabel("Day")
plt.ylabel("Average Sales")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Quantity vs Sales
plt.figure(figsize=(8, 5))
plt.scatter(df["Quantity"], df["Sales"], alpha=0.3)
plt.title("Quantity vs Sales")
plt.xlabel("Quantity")
plt.ylabel("Sales")
plt.show()


# ============================================================
# 7. FEATURE ENGINEERING
# ============================================================

print("\n===== FEATURE ENGINEERING =====")

# Convert Date to datetime
df["Date"] = pd.to_datetime(df["Date"])

# Extract date-related features
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Day_of_Month"] = df["Date"].dt.day

# Create total price before discount
df["Total_Price"] = df["Quantity"] * df["Unit_Price"]

print("\nDate and price features created.")

print(
    df[
        [
            "Date",
            "Year",
            "Month",
            "Day_of_Month",
            "Quantity",
            "Unit_Price",
            "Total_Price",
            "Sales"
        ]
    ].head()
)


# ============================================================
# 8. ENCODING CATEGORICAL FEATURES
# ============================================================

df = pd.get_dummies(
    df,
    columns=[
        "Store_ID",
        "Product_Category",
        "Product",
        "Day"
    ],
    dtype=int
)

# Remove original Date column
df = df.drop("Date", axis=1)

print("\n===== AFTER ENCODING =====")
print("Final dataset shape:", df.shape)


# ============================================================
# 9. SEPARATE FEATURES AND TARGET
# ============================================================

X = df.drop("Sales", axis=1)
y = df["Sales"]

print("\n===== FEATURES AND TARGET =====")
print("X shape:", X.shape)
print("y shape:", y.shape)


# ============================================================
# 10. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\n===== TRAIN / TEST SPLIT =====")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


# ============================================================
# 11. FINAL CONCLUSION
# ============================================================

print("\n===== CONCLUSION =====")

print(
    "The supermarket sales dataset was successfully explored "
    "and prepared for machine learning. EDA was performed to "
    "understand the data structure, distributions, categorical "
    "variables, and relationships with Sales. Feature engineering "
    "was performed by extracting date features and creating "
    "Total_Price. Categorical variables were converted into "
    "numerical features using one-hot encoding. Finally, the "
    "dataset was divided into training and testing sets using "
    "an 80:20 split. The processed data is now ready for the "
    "model-training stage."
)