import pandas as pd
import warnings
warnings.filterwarnings("ignore")
#load the dataset
df = pd.read_csv(r"C:\Users\raghu\OneDrive\Desktop\railway.csv")
print("original dataset:")
print(df.info())
#diplay the dataset first five rows
print("first 5 rows of the dataset:")
print(df.head())
#number of rows and columns
print("\nshape of the dataset:",df.shape)
#describe function guve mean,median,mode ,min,max,25%,50%,75%
print(df.describe())
#handling missing data: check for missing values
missing_data = df.isnull().sum()
print(f"\nmissing data in each column :\n{missing_data}")
# Handling missing values based on column type
# Fill 'Railcard' with a placeholder (assumes missing means "No Railcard")
df['Railcard'].fillna('No Railcard', inplace=True)
# Fill 'Actual Arrival Time' with a placeholder string
df['Actual Arrival Time'].fillna('Not Recorded', inplace=True)
# Fill 'Reason for Delay' with 'No Delay' or 'Unknown'
df['Reason for Delay'].fillna('No Delay', inplace=True)
# Check again for missing values
print("\nMissing values after handling:")
print(df.isnull().sum())

#summary statistic : mean,median,and standard deviation of the price column
print("\nsummary statistic for 'price' column:")
print(f"mean:{df['Price'].mean() .round(2)}")
print(f"median : {df['Price'].median()}")
print(f"standard deviation : {df['Price']. std().round(2)}")
#filtering : filter cars with 'firstClass' Ticket type and count them
ticket_type = df[df['Ticket Class'] == 'First Class']
print(f"\nNumber of rides  with 'First Class' ticket Class: {ticket_type.shape[0]}")
#data aggregation :average price by ticket type
average_price_by_tickettype =  df.groupby('Ticket Class')['Price'].mean()
print(f"\nAverage price by ticket type: \n {average_price_by_tickettype}")
#unique
print("unique values")
print(df['Railcard'].unique())
#count
print(df['Railcard'].value_counts())

import matplotlib.pyplot as plt
plt.figure(figsize=(4,4))
ticket_type_counts = df['Ticket Type'].value_counts()
ticket_type_counts.plot(kind = 'bar', title = "count of rides by Ticket Type" , color = 'red')
plt.ylabel("count")
plt.show()
# Line plot of "Price" for first 20 records
plt.figure(figsize=(8, 6))
plt.plot(df["Price"].head(20), marker='o', linestyle='-', color='blue')
plt.xlabel("Record Index")
plt.ylabel("Price")
plt.title("Line Plot of Price for First 20 Records")
plt.grid(True)
plt.show()
# Create a donut chart for the "Journey Status" column
journey_counts = df["Journey Status"].value_counts()  # count occurrence in Journey Status

plt.figure(figsize=(6, 6))
plt.pie(journey_counts,labels=journey_counts.index,autopct='%1.1f%%',colors=['lightgreen', 'red', 'orange'])  # Add more colors if more statuses exist
plt.gca().add_artist(plt.Circle((0, 0), 0.4, color='white'))  # donut hole
plt.title("Donut Chart of Journey Status")
plt.show()

import seaborn as sns
sns.boxplot(y=df['Price'], color = 'green')
plt.ylabel('Price')
plt.title('Price')
plt.show()
#plot histogram of "credit amount"
plt.figure(figsize=(4, 4))
plt.hist(df["Date of Purchase"],bins = 5, color = 'skyblue', edgecolor = 'black')
plt.xlabel("Date of Purchase")
plt.ylabel("Arrival Destination")
plt.title("Histogram of Ticket")
plt.show()
#pie chart 
plt.figure(figsize=(8, 5))
df["Railcard"].value_counts().plot(kind="pie", autopct='%1.1f%%', startangle=140)
plt.xlabel("Railcard")
plt.ylabel("Count")
plt.title("Pie Chart of Railcard")
plt.xticks(rotation=45)
plt.yticks(rotation=45)
plt.show()
#scatter plot :age vs Credit amount
plt.figure(figsize=(8, 5))
plt.scatter(df["Purchase Type"],df["Price"],color= 'purple', alpha =0.6)
#alpha range from 0(completely transparent) to 1 (compltely opaque)
#alpha = 0.6  means the points are 60% opaque (for 40% transparent)
plt.xlabel("Purchase Type")
plt.ylabel("Price")
plt.title("Scatter plot: Purchase Type vs Price")
plt.show()

#create a liner plot using seaborn
plt.figure(figsize=(8,6))
sns.lineplot(x=df["Departure Time"], y=df["Price"], marker='o', color='green')
plt.xlabel("Departure Time")
plt.ylabel("Price")
plt.title(f"Ticket Price vs Departure Time (seaborn)")
plt.xticks(rotation=45)
plt.show()
plt.figure(figsize=(10, 6))
#bar plot
# Get top 10 departure stations by average ticket price
top_stations = df.groupby("Departure Station")["Price"].mean().nlargest(10)
sns.barplot(x=top_stations.index, y=top_stations.values, palette="coolwarm")
plt.title("Top 10 Departure Stations by Average Ticket Price (Seaborn)", fontsize=14)
plt.xlabel("Departure Station", fontsize=12)
plt.ylabel("Average Ticket Price (£)", fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
#box plot : Ticket Price Distribution by Ticket Type
plt.figure(figsize=(8,5))
sns.boxplot(x="Ticket Type", y="Price", hue="Ticket Type", data=df)
plt.title("Ticket Price Distribution by Ticket Type (Box Plot)")
plt.xlabel("Ticket Type")
plt.ylabel("Ticket Price (£)")
plt.yscale("log")  # For better visualization if prices vary widely
plt.show()
#heatmap
df_encoded = df.copy()
categorical_cols = ['Purchase Type', 'Ticket Class', 'Ticket Type', 'Journey Status', 'Refund Request']
df_encoded[categorical_cols] = df_encoded[categorical_cols].apply(lambda col: col.astype('category').cat.codes)

# Select numeric columns and compute correlation matrix
plt.figure(figsize=(8, 5))
corr_matrix = df_encoded.select_dtypes(include=['number']).corr()

# Plot heatmap
sns.heatmap(corr_matrix, annot=True, linewidths=0.5, cmap='coolwarm')
plt.title("Correlation Heatmap (Seaborn)")
plt.show()
#--------------------------------------------------
#Z -TEST
from statsmodels.stats.weightstats import ztest
import pandas as pd

# The ztest function from statsmodels is used for performing a Z-test to compare
# the means of two independent samples. It tests whether the means of two
# populations are significantly different based on the sample data.

# Select ticket prices for two types of purchase: Online and Station
online_price = df[df["Purchase Type"] == "Online"]["Price"].dropna()
station_price = df[df["Purchase Type"] == "Station"]["Price"].dropna()

# Perform Z-test
z_stat, p_value = ztest(online_price, station_price)

print(f"Z-statistic: {z_stat:.4f}")
print(f"p-value: {p_value:.4f}")

# Interpretation
alpha = 0.05   # significance level
if p_value < alpha:
    print("Reject the null hypothesis: There is a significant difference in ticket prices between Online and Station purchases.")
else:
    print("Fail to reject the null hypothesis: No significant difference in ticket prices.")

#-------------------------------------------------------------
# OUTLIER PROGRAM;
# Select numerical columns (like 'Price')

numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns

# IQR (Interquartile Range) to handle outliers
Q1 = df[numerical_columns].quantile(0.25)
Q3 = df[numerical_columns].quantile(0.75)
IQR = Q3 - Q1

# Define upper and lower bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Identify outliers using IQR
outliers_iqr = (df[numerical_columns] < lower_bound) | (df[numerical_columns] > upper_bound)
print("Outliers detected using IQR method:")
print(outliers_iqr.sum())  # Count of outliers in each column

# Visualization using boxplot (with log scales, no y-label)
plt.figure(figsize=(8, 6))
df[numerical_columns].boxplot(rot=45)
plt.title("Box Plot for Outlier Detection")
plt.xscale("log")
plt.yscale("log")
plt.grid(True)
plt.show()
#---------------------------
#T - TEST
#is proform when the values is > 30;

from scipy.stats import ttest_ind

online_prices = df[df["Purchase Type"] == "Online"]["Price"].dropna()
station_prices = df[df["Purchase Type"] == "Station"]["Price"].dropna()

# Ensure there's enough data for the test
if len(online_prices) > 1 and len(station_prices) > 1:
    # Perform Welch's T-test (assumes unequal variances)
    t_stat, p_value = ttest_ind(online_prices, station_prices, equal_var=False)

    print(f"T-statistic: {t_stat:.4f}")
    print(f"p-value: {p_value:.4f}")

    alpha = 0.05
    if p_value < alpha:
        print("There is a significant difference in ticket prices between Online and Station purchases.")
    else:
        print("No significant difference in ticket prices.")
else:
    print("Not enough data for T-test.")
#-------------------------------
# Z-Score
# Uses mean and standard deviation
from scipy import stats
numerical_columns = df.select_dtypes(include=["float64", "int64"]).columns

# Compute Z-scores
z_scores = stats.zscore(df[numerical_columns].dropna())

# Identify outliers: values beyond ±3 standard deviations
outliers = (abs(z_scores) > 3).sum()
print("Outliers detected using Z-score method:")
print(outliers)

# The Z-score method considers a value an outlier if it is more than 3 std deviations away
# Since the output may be 0, it means no points were extreme enough

# Visualizing distributions with histograms and KDE
for col in numerical_columns:
    sns.histplot(df[col].dropna(), bins=30, kde=True, color="black")
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.show()

# Quick check of distribution shape with a boxplot
for col in numerical_columns:
    sns.boxplot(x=df[col], color="red")
    plt.title(f"Box Plot of {col}")
    plt.show()


