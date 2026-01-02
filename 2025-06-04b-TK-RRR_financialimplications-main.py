#Press strg I to chat with copilot
# ctrl + enter: gitub copilot suggestion
# ctrl + alt + I: open chat view
#debugging F5
#Autoformation Shift+Alt+F
# """ """ for multiline comments
#==============================================================================
# 1. Importing the data files & Data Exploration
#==============================================================================
# Importing the libraries
import pandas as pd
import openpyxl
import numpy as np

###To DO
# add value weighted firm index as benchmark to take care of industry effects 
# add firm descriptive statistics
# maybe change code sections merging before data analysis, as when merging I discovered that I need to drop a firm -> backwards now

# Importing the data files
# Firm Funamentals
file1_path = "C:\\Users\\thkraft\\eCommerce-Goethe Dropbox\\Thilo Kraft\\Thilo(privat)\\Privat\\Research\\RRR_FinancialImplication\\Data\\2025-0319a-TK-fundamentals_Python.xlsx"
# Quarterly Revenue
file2_path = "C:\\Users\\thkraft\\eCommerce-Goethe Dropbox\\Thilo Kraft\\Thilo(privat)\\Privat\\Research\\RRR_FinancialImplication\\Data\\2025-0319a-TK-quarterlyrevenue-collection_Python.xlsx"

try:
    # Read the fundamentals file
    df_fundamentals = pd.read_excel(file1_path, header=None);  # Load with no header
    print("Fundamentals-Daten erfolgreich eingelesen.")
    print("Shape:", df_fundamentals.shape)


    # Read the revenue file
    df_revenue = pd.read_excel(file2_path, header=None);       # Load with no header
    print("\nRevenue-Daten erfolgreich eingelesen.")
    print("Shape:", df_revenue.shape)


except Exception as e:
    #Exception is the base class for all exceptions (FilenotFoundError, ValueError, etc.)
    #As e saves the "exception"
    print(f"Fehler beim Einlesen der Dateien: {e}")


##################################################################
# 1.1 Are there any missing firms in the data files? Compare both#
##################################################################

# Normalize strings to ensure case-insensitive and whitespace-consistent comparison
def normalize_strings(strings):
    return set(s.upper().strip() for s in strings)

# Normalize firm names from the first row of both files and # Extract all unique strings from the first row of both files
normalized_firm_names_fundamentals = normalize_strings(set(df_fundamentals.iloc[0].dropna().astype(str).str.strip()));
normalized_firm_names_revenue = normalize_strings(set(df_revenue.iloc[0].dropna().astype(str).str.strip()));

# Compare normalized firm names
missing_in_revenue = normalized_firm_names_fundamentals - normalized_firm_names_revenue;
missing_in_fundamentals = normalized_firm_names_revenue - normalized_firm_names_fundamentals;

# Print the number of unique firms in each dataset
print(f"Number of Unique Firms in Fundamentals File: {len(normalized_firm_names_fundamentals)}")
print(f"Number of Unique Firms in Revenue File: {len(normalized_firm_names_revenue)}")

# Print results
print("Firm Names Missing in Revenue File:")
print(missing_in_revenue)

print("\nFirm Names Missing in Fundamentals File:")
print(missing_in_fundamentals)

#==============================================================================
# 2. Data Wrangling
#==============================================================================

###################
# 2.1 Revenue Data#
###################

# Extract the first two rows for processing headers
header_rows_rev = df_revenue.iloc[:2];
data_rows_rev = df_revenue.iloc[2:];  # Remaining rows are the actual data

# Combine the first two rows to create a single-level column header
combined_headers = header_rows_rev.apply(lambda x: x.str.strip() if x.dtype == "object" else x);
column_headers = combined_headers.apply(lambda x: '.'.join(x.dropna()), axis=0);
data_rows_rev.columns = column_headers;  # Assign proper headers to the data rows

# Rename the first column to "Date"
data_rows_rev.rename(columns={data_rows_rev.columns[0]: 'Date'}, inplace=True);

# Reset the index for the data rows
df_revenue = data_rows_rev.reset_index(drop=True);

### replace empty cells with NaN
df_revenue = df_revenue.replace(r'^\s*$', pd.NA, regex=True);



#Add totel and RRR and shares
# Remove duplicate columns
df_revenue = df_revenue.loc[:, ~df_revenue.columns.duplicated()];
# Identify columns related to "New Customer" and "Returning Customer"
new_customer_cols = [col for col in df_revenue.columns if '#New_Customers' in col];
returning_customer_cols = [col for col in df_revenue.columns if '#Returning_Customers' in col];

"""
code is slow not optimal-> try to improve
"""

# Create new columns for Total Revenue, share, RRR, revenue growth, and revenue_new_growth
for new_col, ret_col in zip(new_customer_cols, returning_customer_cols):
    # Generate new column names
    total_revenue_col = new_col.replace('#New_Customers', '#Total_Revenue')
    share_ret_revenue_col = ret_col.replace('#Returning_Customers', '#Share_Ret_Revenue')
    rrr_col = ret_col.replace('#Returning_Customers', '#RRR')
    revenue_growth_col = ret_col.replace('#Returning_Customers', '#Revenue_Growth')
    acq_rate_col = new_col.replace('#New_Customers', '#Acq_Rate')
    gm_col = new_col.replace('#New_Customers', '#Growth_Mix')
    growth_indicator_col = new_col.replace('#New_Customers', '#Growth_Indicator')

    # Add Total Revenue
    df_revenue[total_revenue_col] = df_revenue[new_col] + df_revenue[ret_col]

    # Calculate Share of Returning Revenue
    #### have some 0 df_revenue[share_ret_revenue_col] = df_revenue[ret_col] / df_revenue[total_revenue_col].replace(0, np.nan)
    df_revenue[share_ret_revenue_col] = df_revenue[ret_col] / df_revenue[total_revenue_col].replace(0, np.nan)

    # Calculate RRR
    df_revenue[rrr_col] = df_revenue[ret_col] / df_revenue[total_revenue_col].shift(1).replace(0, np.nan)

    # Calculate Revenue Growth
    df_revenue[revenue_growth_col] = (
        (df_revenue[total_revenue_col] - df_revenue[total_revenue_col].shift(1)) / df_revenue[total_revenue_col].shift(1)
    )

    # Calculate Revenue New Growth
    df_revenue[acq_rate_col] = (df_revenue[new_col] / df_revenue[new_col].shift(1))

    # Calculate Growth Mix
    df_revenue[gm_col] = (df_revenue[ret_col] -  df_revenue[ret_col].shift(1)) / (df_revenue[total_revenue_col] -  df_revenue[total_revenue_col].shift(1))

    # calculate Growth Indicator
    df_revenue[growth_indicator_col] = df_revenue[revenue_growth_col].apply(
        lambda x: 1 if x > 0 else (-1 if x < 0 else 0)
    )

    # Reorder columns to place new columns next to Returning Customers
    cols = list(df_revenue.columns) #Get current column order
    ret_col_index = cols.index(ret_col) # Find index of the Returning Customer column
    # Remove and reinsert new columns next to the Returning Customer column
    cols.remove(total_revenue_col)
    cols.remove(share_ret_revenue_col)
    cols.remove(rrr_col)
    cols.remove(revenue_growth_col)
    cols.remove(acq_rate_col)
    cols.remove(gm_col)
    cols.remove(growth_indicator_col)

    cols.insert(ret_col_index + 1, total_revenue_col)
    cols.insert(ret_col_index + 2, share_ret_revenue_col)
    cols.insert(ret_col_index + 3, rrr_col)
    cols.insert(ret_col_index + 4, revenue_growth_col)
    cols.insert(ret_col_index + 5, acq_rate_col)
    cols.insert(ret_col_index + 6, gm_col)
    cols.insert(ret_col_index + 7, growth_indicator_col)
    df_revenue = df_revenue[cols]

        

#######################
# 2.2 Fundamental Data#
#######################

"""
I have some missing cells or 0 which I need to replace with 0 like for R&D expenses
some variables are named the same as in 2.1
"""
### Rename Headers
### Rename Headers
### Rename Headers
# Step 1: Process Headers
header_rows_fun = df_fundamentals.iloc[:2];  # First two rows as headers
data_rows_fun = df_fundamentals.iloc[2:];  # Remaining rows are data

# Combine headers: "Firm.Variable"
combined_headers = header_rows_fun.apply(lambda x: x.astype(str).str.strip(), axis=0);
column_headers = combined_headers.apply(lambda x: '.'.join(x.dropna()), axis=0);
data_rows_fun.columns = column_headers;  # Assign proper headers to the data rows
df_fundamentals = data_rows_fun.reset_index(drop=True);  # Reset index with proper columns

# Step 2: Handle Missing Data
df_fundamentals.replace('#N/A N/A', np.nan, inplace=True)
# Remove duplicate columns
df_fundamentals = df_fundamentals.loc[:, ~df_fundamentals.columns.duplicated()];

### Check for missing values
### Check for missing values
### Check for missing values
# Extract variable names (after the dot)
variable_names = [col.split('.')[-1] for col in df_fundamentals.columns];

# Compute missing values per variable
missing_counts = df_fundamentals.isnull().sum();
missing_summary = pd.DataFrame({'Variable': variable_names, 'Missing_Count': missing_counts.values});

# Aggregate by variable name
missing_summary_grouped = missing_summary.groupby('Variable')['Missing_Count'].sum().sort_values(ascending=False);

# Display the top variables with the most missing values
print("Top variables with missing values:")
print(missing_summary_grouped.head(10))  # Show top 10 variables with most missing values

# Replace NaN values with 0 in the selected columns
columns_to_replace = [col for col in df_fundamentals.columns if "IS_RD_EXPEND" in col or "IS_SGA_EXPENSE" in col]
df_fundamentals[columns_to_replace] = df_fundamentals[columns_to_replace].fillna(0)


# Extract firm names (before the dot in "Firm.Variable")
firm_names = [col.split('.')[0] for col in df_fundamentals.columns];

# Create a DataFrame with firm names and their missing values
missing_summary_firms = pd.DataFrame({'Firm': firm_names, 'Missing_Count': missing_counts.values});

# Aggregate missing values per firm
missing_summary_firms_grouped = missing_summary_firms.groupby('Firm')['Missing_Count'].sum().sort_values(ascending=False);

# Display the top firms with the most missing values
print("Top firms with missing values:")
print(missing_summary_firms_grouped.head(10))  # Show top 10 firms with most missing values



#==============================================================================
# 3. First Analysis of seperated data
#==============================================================================

###################
# 3.1 Revenue Data#
###################

import seaborn as sns
import matplotlib.pyplot as plt

"""
make it more efficient, create function
"""


# Define the columns of interest
columns_of_interest = ["#RRR", "#Revenue_Growth", "#Acq_Rate", "#Share_Ret_Revenue", "#Growth_Mix"]

# Ensure the Date column is in datetime format for proper sorting
df_revenue['Date'] = pd.to_datetime(df_revenue['Date'])

# Iterate over each metric to generate boxplots and histograms
for metric in columns_of_interest:
    # Extract relevant columns for the current metric
    metric_cols = [col for col in df_revenue.columns if metric in col]

    # Melt the dataset for easier plotting with seaborn
    df_metric_melted = df_revenue.melt(
        id_vars=['Date'], 
        value_vars=metric_cols, 
        var_name='Firm', 
        value_name=metric
    )

    # Remove the top 4 highest values correctly for display purposes (not statistics)
    df_filtered = df_metric_melted.copy()
    df_filtered = df_filtered[df_filtered[metric] < df_filtered[metric].nlargest(4).min()]  # Exclude top 4 values
    # Remove the lowest 4 values for display purposes (not statistics)
    df_filtered = df_filtered[df_filtered[metric] > df_filtered[metric].nsmallest(4).max()]  # Exclude lowest 4 values

    # Compute additional statistics
    q25, q50, q75 = df_metric_melted[metric].quantile([0.25, 0.50, 0.75])

    # Ensure the x-axis (dates) remain sorted
    df_filtered = df_filtered.sort_values(by="Date")

    # Boxplot
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Date', y=metric, data=df_filtered, color='skyblue')

    # Customize the chart
    plt.xlabel('Date')
    plt.ylabel(metric.replace("#", ""))  # Remove '#' for cleaner labels
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

    # Print summary statistics including quantiles
    print(f"\nSummary statistics for {metric.replace('#', '')}:")
    print(df_metric_melted[metric].describe().loc[['mean', 'std', 'min', 'max']])
    print(f"25th Percentile (Q1): {q25:.2f}")
    print(f"50th Percentile (Q2): {q50:.2f}")
    print(f"75th Percentile (Q3): {q75:.2f}")

    # Histogram
    plt.figure(figsize=(10, 6))
    sns.histplot(df_filtered[metric], bins=30, kde=True, color='skyblue')

    # Customize the plot
    plt.xlabel(metric.replace("#", ""))
    plt.ylabel('Frequency')
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.show()

"""
Summary statistics for RRR (Excluding Top 4):
mean    0.799884
std     0.312226
min     0.001062
max     3.889442
Name: #RRR, dtype: float64
25th Percentile (Q1): 0.61
75th Percentile (Q3): 0.95

Summary statistics for Revenue_Growth (Excluding Top 4):
mean    0.080728
std     0.410411
min    -0.998914
max     5.636012
Name: #Revenue_Growth, dtype: float64
25th Percentile (Q1): -0.08
75th Percentile (Q3): 0.17

Summary statistics for Revenue_New_Growth:
mean     0.287643
std      0.411074
min      0.000024
max     16.130858
Name: #Acq_Rate, dtype: float64
25th Percentile (Q1): 0.10
75th Percentile (Q3): 0.37
"""


### Create RRR bins and calculate average growth rate
### Create RRR bins and calculate average growth rate
### Create RRR bins and calculate average growth rate

# Extract company names by identifying unique prefixes before the dot (".")
company_names = set([col.split('.')[0] for col in df_revenue.columns if '.' in col])

# Define bins (quartiles) for RRR within each period
bin_labels = ["Q1 (Lowest 25%)", "Q2", "Q3", "Q4 (Highest 25%)"]

# List to store period-wise bin averages
bin_avg_growth_list = []
bin_avg_new_growth_list = [] #store period-wise new growth averages


# Sort dataset by Date
df_revenue = df_revenue.sort_values(by="Date")

# Identify the first available period (which has no RRR)
first_period = df_revenue["Date"].min()


# NEW: generic helper – bin by any metric each period, average a target, then average across periods
def period_quantile_median(df_revenue, company_names, date_col="Date",
                          sort_metric_tag="#RRR", target_metric_tag="#Revenue_Growth",
                          q=4, labels=None, min_firms=4, lead=0):
    """
    lead: int, number of periods to shift target forward (positive = future values)
          e.g., lead=1 means use next period's target value
    """
    labels = labels or [f"Q{i}" for i in range(1, q+1)]
    bin_avg_list = []
    
    # Sort by date to ensure proper ordering
    df_sorted = df_revenue.sort_values(by=date_col).reset_index(drop=True)
    dates = df_sorted[date_col].unique()
    
    for i, date in enumerate(dates):
        # Skip if we can't look ahead far enough
        if lead > 0 and i + lead >= len(dates):
            continue
            
        group = df_sorted[df_sorted[date_col] == date]
        
        # If lead > 0, get target values from future period
        if lead > 0:
            target_date = dates[i + lead]
            target_group = df_sorted[df_sorted[date_col] == target_date]
        else:
            target_group = group
        
        rows = []
        for company in company_names:
            s_col = f"{company}.{sort_metric_tag}"
            t_col = f"{company}.{target_metric_tag}"
            
            # Get sort value from current period
            if s_col in group.columns:
                s_val = pd.to_numeric(group[s_col].values[0], errors="coerce")
            else:
                continue
                
            # Get target value from current or future period
            if t_col in target_group.columns:
                t_val = pd.to_numeric(target_group[t_col].values[0], errors="coerce")
            else:
                continue
                
            if pd.notna(s_val) and pd.notna(t_val):
                rows.append((company, s_val, t_val))
                
        if len(rows) < min_firms:
            continue

        df_f = pd.DataFrame(rows, columns=["Company", "SORT", "TARGET"])
        try:
            df_f["Q"] = pd.qcut(df_f["SORT"], q=q, labels=labels, duplicates="drop")
        except ValueError:
            continue

        by_q = df_f.groupby("Q", observed=True)["TARGET"].median().reindex(labels)
        out = by_q.to_frame().T
        out["Date"] = date
        bin_avg_list.append(out)

    df_bin_avg = pd.concat(bin_avg_list, ignore_index=True) if bin_avg_list else pd.DataFrame(columns=labels+["Date"])
    final_avg = df_bin_avg.drop(columns="Date", errors="ignore").median(numeric_only=True)
    return df_bin_avg, final_avg

# Current period RRR → Current period Revenue Growth
bin_by_RRR, final_avg_growth_by_RRR = period_quantile_median(
    df_revenue, company_names,
    sort_metric_tag="#RRR",
    target_metric_tag="#Revenue_Growth",
    q=5, labels=["Q5","Q4","Q3","Q2", "Q1"],
    lead=0  # same period
)
print("Current Period RRR → Current Growth:")
print(final_avg_growth_by_RRR)



# Current period Acq_Rate
bin_by_Acq_lead0, final_avg_growth_by_Acq_lead0 = period_quantile_median(
    df_revenue, company_names,
    sort_metric_tag="#Acq_Rate",
    target_metric_tag="#Revenue_Growth",
    q=5, labels=["Q1","Q2","Q3","Q4", "Q5"],
    lead=0  # same period
)
print("\nCurrent Period Acq Rate → Current Growth")
print(final_avg_growth_by_Acq_lead0)


# Current period RRR → NEXT period Revenue Growth (forecast)
bin_by_RRR_lead1, final_avg_growth_by_RRR_lead1 = period_quantile_median(
    df_revenue, company_names,
    sort_metric_tag="#RRR",
    target_metric_tag="#Revenue_Growth",
    q=5, labels=["Q1","Q2","Q3","Q4", "Q5"],
    lead=1  # next period
)
print("\nCurrent Period RRR → Next Period Growth (Lead=1):")
print(final_avg_growth_by_RRR_lead1)

# Current period Acq_Rate → NEXT period Revenue Growth
bin_by_Acq_lead1, final_avg_growth_by_Acq_lead1 = period_quantile_median(
    df_revenue, company_names,
    sort_metric_tag="#Acq_Rate",
    target_metric_tag="#Revenue_Growth",
    q=5, labels=["Q1","Q2","Q3","Q4", "Q5"],
    lead=1  # next period
)
print("\nCurrent Period Acq Rate → Next Period Growth (Lead=1):")
print(final_avg_growth_by_Acq_lead1)


# Current period RRR → four period Revenue Growth

bin_by_RRR_lead4, final_avg_growth_by_RRR_lead4 = period_quantile_median(
    df_revenue, company_names,
    sort_metric_tag="#RRR",
    target_metric_tag="#Revenue_Growth",
    q=5, labels=["Q1","Q2","Q3","Q4", "Q5"],
    lead=4  # four periods ahead
)
print("\nCurrent Period RRR → Four Periods Ahead Growth (Lead=4):")
print(final_avg_growth_by_RRR_lead4)

# Current period Acq_Rate → four period Revenue Growth
bin_by_Acq_lead4, final_avg_growth_by_Acq_lead4 = period_quantile_median(
    df_revenue, company_names,
    sort_metric_tag="#Acq_Rate",
    target_metric_tag="#Revenue_Growth",
    q=5, labels=["Q1","Q2","Q3","Q4", "Q5"],
    lead=4  # four periods ahead
)
print("\nCurrent Period Acq Rate → Four Periods Ahead Growth (Lead=4):")
print(final_avg_growth_by_Acq_lead1)




"""
really only current period -> in Q1 seems an outliner
"""

#==============================================================================
# 4. Merging the Data
#==============================================================================

# Exclude specific columns from df_revenue
excluded_columns = ['Date'];
df_revenue_to_append = df_revenue.drop(columns=excluded_columns, errors='ignore');

df_fundamentals = df_fundamentals.rename(columns={'nan.Dates': 'Date'});
df_combined = pd.concat([df_fundamentals, df_revenue_to_append], axis=1);

# Convert all column headers to uppercase
df_combined.columns = df_combined.columns.str.upper()
# Remove duplicate columns by keeping the first occurrence
df_combined = df_combined.loc[:, ~df_combined.columns.duplicated()]

# Identify all variables except "Date"
all_vars = [col for col in df_combined.columns if col not in ["DATE"]]

# Melt the dataset to create a long format
df_long = df_combined.melt(
    id_vars=["DATE"],  # Keep the Date column as an identifier
    value_vars=all_vars,  # Include all variables (original + calculated)
    var_name="Firm_Variable",  # Combined Firm and Variable column
    value_name="Value"  # Values for all variables
)

# Split "Firm_Variable" into "Firm" and "Variable"
df_long[['FIRM', 'VARIABLE']] = df_long['Firm_Variable'].str.rsplit('.', n=1, expand=True)

# Pivot the dataset to create separate columns for variables
df_long = df_long.pivot_table(
    index=['FIRM', 'DATE'],  # Group by Firm and Date
    columns='VARIABLE',  # Create columns for variables
    values='Value',  # Values to fill
).reset_index()

# Set the MultiIndex for panel regression
df_long = df_long.set_index(['FIRM', 'DATE'])


# Test the revenue data
print(f"Correlation between merged datasets is: {df_long['#TOTAL_REVENUE'].corr(df_long['SALES_REV_TURN']):.2f}")
#Correlation between merged datasets is: 0.94

# Share of #Total_Revenue of SALES_REV_TURN in % 
df_long['TOTAL_REVENUE_PCT_SALES_REV_TURN'] = (df_long['#TOTAL_REVENUE'] / df_long['SALES_REV_TURN'].replace(0, np.nan)) / 1000000 * 100 # /1000000 to account for unit difference * 100 for percentage
# Create the boxplot
plt.figure(figsize=(6, 4))
plt.boxplot(df_long['TOTAL_REVENUE_PCT_SALES_REV_TURN'], vert=True)
plt.title('Boxplot of Share of observed revenue to reported revenue')
plt.ylabel('Percentage')
plt.grid(True)
plt.show()
print(f"Average share of reported revenue in observed revenue: {df_long['TOTAL_REVENUE_PCT_SALES_REV_TURN'].mean():.2f}%")
# Average share of reported revenue in observed revenue: 4.13 % 

'''
I have shares of outliners above 100%-> cant be
# 1) Create the mask for ratios above 100%
mask = df_long['TOTAL_REVENUE_PCT_SALES_REV_TURN'] > 60

# 2) Extract and view those rows
outliers = df_long.loc[mask].reset_index()
print(f"Found {len(outliers)} rows with ratio > 100%:")
print(outliers[['FIRM', 'DATE', 'TOTAL_REVENUE_PCT_SALES_REV_TURN']])

VARIABLE            FIRM       DATE TOTAL_REVENUE_PCT_SALES_REV_TURN
0         CART US EQUITY 2019-03-29                       109.350886
1         CART US EQUITY 2019-06-28                       104.114378
manually check in the data

Data points in ALTD: 44833863.1, 47892613.98
Data points in Fund: 41 46

if i lower the threshold to 60 it seems to only concern the CART US EQUITY
-> exclude CART US Equity
-> exclude it in the alt D data already in case it screws the revenue portfolio analysis

'''


### Add needed columns to the dataset
### Add needed columns to the dataset
### Add needed columns to the dataset

# 1) Estimate new and retained revenue by scaling with the total‐revenue ratio
df_long['NEW_REV_EST']   = df_long['#NEW_CUSTOMERS']      / (df_long['TOTAL_REVENUE_PCT_SALES_REV_TURN'] / 100) / 1000000
df_long['RETAINED_REV_EST']   = df_long['#RETURNING_CUSTOMERS'] / (df_long['TOTAL_REVENUE_PCT_SALES_REV_TURN'] / 100) / 1000000 #to account for uni displayment F Data is displayed in millions

# 2) Sum to get an estimated total revenue
df_long['SALES_REV_TURN_EST'] = df_long['RETAINED_REV_EST'] + df_long['NEW_REV_EST']

#Corr is one by contruction

'''
#Calculate Retained & New Revenue Estimate
df_long['RETAINED_REV_EST'] = df_long['#RRR'] * df_long['SALES_REV_TURN'].shift(1) # in million


df_long['NEW_REV_EST'] = df_long['SALES_REV_TURN'] - df_long['RETAINED_REV_EST'] # in million
# Check if any NEW_REV_EST values are negative
if (df_long['NEW_REV_EST'] < 0).any():
    print("There are rows with NEW_REV_EST < 0.")
else:
    print("No rows with NEW_REV_EST < 0.")


what do you do when est new rev < 0 ? ->nothing? estimation error
# 1) Create a boolean mask for negative new revenue estimates
mask_neg_new_rev = df_long['NEW_REV_EST'] < 0

# 2) View all rows where NEW_REV_EST is negative
negative_new_rev = df_long.loc[mask_neg_new_rev]

# 3) Optionally, display only the key columns (Firm, Date, NEW_REV_EST)
print(negative_new_rev.reset_index()[['FIRM', 'DATE', 'NEW_REV_EST']])
400 rows


df_long['SALES_REV_TURN_EST'] = df_long['RETAINED_REV_EST'] + df_long['NEW_REV_EST'] # in million

print(f"Correlation between SALES REV TURN and the Estimated through RRR and lag is: {df_long['SALES_REV_TURN_EST'].corr(df_long['SALES_REV_TURN']):.2f}")
# Correlation between SALES REV TURN and the Estimated through RRR and lag is: 1.0000 by definition though

print(f"Correlation between #New Customer and the Estimated New Revenue is: {df_long['#NEW_CUSTOMERS'].corr(df_long['NEW_REV_EST']):.2f}")
# Correlation between SALES REV TURN and the Estimated through RRR and lag is: 0.3

BiG Problem

print(f"Correlation between #Returning Customer and the Estimated Returning Revenue is: {df_long['#RETURNING_CUSTOMERS'].corr(df_long['RETAINED_REV_EST']):.2f}")
# Correlation between SALES REV TURN and the Estimated through RRR and lag is: 0.96

'''

#Calculate Metrics
#Calculate Metrics
#Calculate Metrics

#RRR
df_long['RRR_PCT'] = df_long['#RRR'] * 100

df_long['RRR_PCT_LAG1'] = df_long.groupby(level='FIRM')['RRR_PCT'].shift(1)
df_long['RRR_LAG'] = df_long['RRR_PCT_LAG1'] / 100
df_long['RRR_PCT_LAG2'] = df_long.groupby(level='FIRM')['RRR_PCT'].shift(2)
df_long['RRR_PCT_LAG3'] = df_long.groupby(level='FIRM')['RRR_PCT'].shift(3)
df_long['RRR_PCT_LAG4'] = df_long.groupby(level='FIRM')['RRR_PCT'].shift(4)
#Acq Rate
df_long['ACQ_RATE_PCT'] = df_long['#ACQ_RATE'] * 100
df_long['ACQ_RATE_PCT_LAG1'] = df_long.groupby(level='FIRM')['ACQ_RATE_PCT'].shift(1)
df_long['ACQ_RATE_PCT_LAG2'] = df_long.groupby(level='FIRM')['ACQ_RATE_PCT'].shift(2)
df_long['ACQ_RATE_PCT_LAG3'] = df_long.groupby(level='FIRM')['ACQ_RATE_PCT'].shift(3)
df_long['ACQ_RATE_PCT_LAG4'] = df_long.groupby(level='FIRM')['ACQ_RATE_PCT'].shift(4)

# "CLV+"
df_long['RET_REV_ACQ_LAG'] = df_long['NEW_REV_EST'].shift(1) * df_long['RRR_PCT'] / 100 * df_long['PM_OPER']
df_long['RET_REV_ACQ_LAG2'] = df_long['NEW_REV_EST'].shift(2) * df_long['RRR_PCT'] / 100 * df_long['RRR_PCT_LAG1'] / 100 * df_long['PM_OPER']
df_long['RET_REV_ACQ_LAG3'] = df_long['NEW_REV_EST'].shift(3) * df_long['RRR_PCT'] / 100 * df_long['RRR_PCT_LAG1'] / 100 * df_long['RRR_PCT_LAG2'] / 100 * df_long['PM_OPER']
df_long['RET_REV_ACQ_LAG4'] = df_long['NEW_REV_EST'].shift(4) * df_long['RRR_PCT'] / 100 * df_long['RRR_PCT_LAG1'] / 100 * df_long['RRR_PCT_LAG2'] / 100 * df_long['RRR_PCT_LAG3'] / 100 * df_long['PM_OPER']


#IS_SGA_EXPENSE_PCT
df_long['IS_SGA_EXPENSE_PCT'] = (df_long['IS_SGA_EXPENSE'] / df_long['SALES_REV_TURN'].replace(0, np.nan)) * 100

#Revenue Growth 
df_long['SALES_REV_TURN_LAG1'] = (
    df_long
      .groupby(level='FIRM')['SALES_REV_TURN']
      .shift(1)
)
df_long['REV_GROWTH_PCT'] = (np.log(df_long['SALES_REV_TURN'].replace(0, np.nan))- np.log(df_long['SALES_REV_TURN_LAG1'].replace(0, np.nan)))*100 
df_long.drop(columns='SALES_REV_TURN_LAG1', inplace=True)

#Profit Growth 
df_long['IS_OPER_INC_LAG1'] = df_long.groupby(level='FIRM')['IS_OPER_INC'].shift(1)
df_long['IS_OPER_INC_GROWTH_PCT'] = (df_long['IS_OPER_INC'] - df_long['IS_OPER_INC_LAG1']) / df_long['IS_OPER_INC_LAG1'].replace(0, np.nan) * 100
df_long['IS_OPER_INC_GROWTH'] = df_long['IS_OPER_INC_GROWTH_PCT'] / 100

# OI Profit Margin
df_long['PM_OPER_PCT'] = (df_long['IS_OPER_INC'] / df_long['SALES_REV_TURN'].replace(0, np.nan)) * 100
df_long['PM_OPER'] = (df_long['IS_OPER_INC'] / df_long['SALES_REV_TURN'].replace(0, np.nan)) 


# Other Metrics
df_long['RET_EST_X_SHARE_RET'] = df_long['RETAINED_REV_EST'] * df_long['#SHARE_RET_REVENUE']
df_long['NEW_REV_GROWTH'] = ((df_long['ACQ_RATE_PCT']) * (1-df_long['#SHARE_RET_REVENUE'].shift(1))).replace(0, np.nan)



df_long['EPR'] = (df_long['IS_OPER_INC'] / df_long['SALES_REV_TURN'].replace(0, np.nan))
df_long['EPS'] = (df_long['IS_OPER_INC'] / df_long['BS_SH_OUT'].replace(0, np.nan))
df_long['REV_MULTIPLE'] = df_long['HISTORICAL_MARKET_CAP'] / df_long['SALES_REV_TURN'].replace(0, np.nan)




#Calculate stock returns
#Calculate stock returns
#Calculate stock returns
# Make sure PX_LAST is numeric
df_long['PX_LAST'] = pd.to_numeric(df_long['PX_LAST'], errors='coerce')

# 2) Compute lag‐1 prices per firm
df_long['PX_LAST_LAG1'] = (
    df_long
      .groupby(level='FIRM')['PX_LAST']
      .shift(1)
)

# 3) Raw log-return (decimal)
df_long['RETURN_LOG'] = np.log(df_long['PX_LAST'] / df_long['PX_LAST_LAG1']) # compunded decimal return over period; time series analysis
df_long['RET_ARITH']  = np.exp(df_long['RETURN_LOG']) - 1 # decimal percentage change over period, not additive, actual percentage change
# 4) Drop the helper column
#df_long.drop(columns='PX_LAST_LAG1', inplace=True)


# Calculate risk-free 
### make sure RF is converted into fitting period returns decimal (e.g quarterly, monthly)
### make sure RF is converted into fitting period returns (e.g quarterly, monthly)
### make sure RF is converted into fitting period returns (e.g quarterly, monthly)
rf_ann_pct = df_long.xs('USBMMY3M INDEX', level='FIRM')['PX_LAST']  # annual return in percent
conversion = 4 #for quarterly
rf = ((1 + rf_ann_pct/100)**(1/conversion) - 1) # risk free rate per period return decimal
dates = df_long.index.get_level_values('DATE')
df_long['RF']     = rf.reindex(dates).values


# Calculate market premiums using log-returns
mkt_log   = df_long.xs('SPX INDEX',    level='FIRM')['RETURN_LOG']
mkt_arith = np.exp(mkt_log) - 1
df_long['MKT_RF'] = mkt_arith.reindex(dates).values - df_long['RF']
df_long['MKT_RF'] = pd.to_numeric(df_long['MKT_RF'], errors='coerce')
df_long['MKT_RF_LAG'] = df_long.groupby(level='FIRM')['MKT_RF'].shift(1)

# 3) Lagged firm characteristics: SIZE, BTM and existing RRR_pct_lag1
df_long['HISTORICAL_MARKET_CAP'] = pd.to_numeric(df_long['HISTORICAL_MARKET_CAP'], errors='coerce')
size = np.log(df_long['HISTORICAL_MARKET_CAP'])
df_long['SIZE'] = size
df_long['SIZE_LAG'] = size.groupby(level='FIRM').shift(1)
btm  = (df_long['BS_TOT_ASSET'] - df_long['BS_TOT_LIAB2']) / df_long['HISTORICAL_MARKET_CAP']
df_long['BTM_LAG'] = btm.groupby(level='FIRM').shift(1)
df_long['BTM'] = btm



# 4) Excess return calculation
df_long['EXCESS_RET'] = df_long['RET_ARITH'] - df_long['RF']


# download the data
# Save the merged dataset to a new Excel file
output_file = "C:\\Users\\thkraft\\eCommerce-Goethe Dropbox\\Thilo Kraft\\Thilo(privat)\\Privat\\Research\\RRR_FinancialImplication\\Data\\df_final-20250315.xlsx"
df_long.to_excel(output_file, index=False)
print(f"Data successfully saved to: {output_file}")
print(f"Number of Firms: {df_long.index.get_level_values('FIRM').nunique()}")


# =============================================================================
# Summary Statistics Table (All Firms, All Periods)
# =============================================================================

print("\n" + "="*80)
print("SUMMARY STATISTICS - ALL FIRMS & PERIODS")
print("="*80)

# Define the variables to summarize
summary_vars = {
    'Assets': 'BS_TOT_ASSET',
    'Market_Value': 'HISTORICAL_MARKET_CAP',
    'Revenue': 'SALES_REV_TURN',
    'RRR (%)': 'RRR_PCT',
    'Acq_Rate (%)': 'ACQ_RATE_PCT',
    'Growth_Mix': '#GROWTH_MIX',
    'Rev_Growth (%)': 'REV_GROWTH_PCT',
    'BTM': 'BTM',
    'Book_Value': 'BS_TOT_ASSET',  # Using total assets as proxy for book value
    'PM (%)': 'PM_OPER_PCT',
    'CF_Growth (%)': 'IS_OPER_INC_GROWTH_PCT',
    'REV_GROWTH_PCT': 'REV_GROWTH_PCT'
}

# Create summary statistics
summary_stats = []

for display_name, col_name in summary_vars.items():
    if col_name in df_long.columns:
        data = pd.to_numeric(df_long[col_name], errors='coerce').dropna()
        
        stats = {
            'Variable': display_name,
            'N_Obs': len(data),
            'Mean': data.mean(),
            'StdDev': data.std(),
            'P25': data.quantile(0.25),
            'Median': data.quantile(0.50),
            'P75': data.quantile(0.75),
            'Min': data.min(),
            'Max': data.max()
        }
        summary_stats.append(stats)
    else:
        print(f"⚠️ Warning: Column '{col_name}' not found in df_long")

# Create DataFrame
summary_table = pd.DataFrame(summary_stats)

# Format numbers for display
summary_table_formatted = summary_table.copy()
for col in ['Mean', 'StdDev', 'P25', 'Median', 'P75', 'Min', 'Max']:
    summary_table_formatted[col] = summary_table_formatted[col].map(lambda x: f"{x:,.2f}")
summary_table_formatted['N_Obs'] = summary_table_formatted['N_Obs'].map(lambda x: f"{x:,}")

print("\n" + summary_table_formatted.to_string(index=False))

# Export to Excel (optional)
output_path = "C:\\Users\\thkraft\\eCommerce-Goethe Dropbox\\Thilo Kraft\\Thilo(privat)\\Privat\\Research\\RRR_FinancialImplication\\Data\\summary_statistics.xlsx"
summary_table.to_excel(output_path, index=False)
print(f"\n✅ Summary statistics exported to: {output_path}")


#==============================================================================
# 5. Regression Analysis
#==============================================================================

### Correlations
### Correlations
### Correlations

def analyze_firm_correlations(df_long, var1='RRR_PCT', var2='RRR_PCT_LAG1', min_obs=3):
    """
    Calculate and visualize correlation between two variables for each firm.
    Parameters:
    -----------
    df_long : DataFrame with MultiIndex ['FIRM', 'DATE']
    var1 : str, first variable name (default 'RRR_PCT')
    var2 : str, second variable name (default 'RRR_PCT_LAG1')
    min_obs : int, minimum observations required per firm (default 3)
    
    Returns:
    --------
    corr_df : DataFrame with correlation results per firm
    """
    print("\n" + "="*80)
    print(f" Correlation between {var1} and {var2}")
    print("="*80)
    
    # Calculate correlation for each firm
    correlations = []
    for firm in df_long.index.get_level_values('FIRM').unique():
        firm_data = df_long.xs(firm, level='FIRM')[[var1, var2]].dropna()
        
        if len(firm_data) >= min_obs:
            corr = firm_data[var1].corr(firm_data[var2])
            correlations.append({
                'FIRM': firm,
                'Correlation': corr,
                'N_Obs': len(firm_data)
            })
    
    # Create DataFrame with results
    corr_df = pd.DataFrame(correlations)
    
    if corr_df.empty:
        print("No firms with sufficient observations found.")
        return corr_df
    
    # Calculate summary statistics
    mean_corr = corr_df['Correlation'].mean()
    median_corr = corr_df['Correlation'].median()
    #std_corr = corr_df['Correlation'].std()
    
    print(f"\nSummary Statistics:")
    print(f"Mean Correlation:   {mean_corr:.4f}")
    print(f"Median Correlation: {median_corr:.4f}")
    print(f"Number of Firms:    {len(corr_df)}")
    
    print(f"\nDistribution:")
    print(f"Min:  {corr_df['Correlation'].min():.4f}")
    print(f"25%:  {corr_df['Correlation'].quantile(0.25):.4f}")
    print(f"50%:  {corr_df['Correlation'].quantile(0.50):.4f}")
    print(f"75%:  {corr_df['Correlation'].quantile(0.75):.4f}")
    print(f"Max:  {corr_df['Correlation'].max():.4f}")
    
    # Histogram of correlations
    plt.figure(figsize=(10, 6))
    plt.hist(corr_df['Correlation'], bins=30, edgecolor='black', alpha=0.7)
    plt.axvline(mean_corr, color='red', linestyle='--', linewidth=2, 
                label=f'Mean = {mean_corr:.3f}')
    plt.axvline(median_corr, color='green', linestyle='--', linewidth=2, 
                label=f'Median = {median_corr:.3f}')
    plt.xlabel(f'Correlation ({var1} vs {var2})')
    plt.ylabel('Number of Firms')
    plt.title(f'Distribution of Correlation Across Firms\n({var1} vs {var2})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return corr_df



# LAG1 persistence
corr_lag1 = analyze_firm_correlations(df_long, var1='RRR_PCT', var2='RRR_PCT_LAG1')

# LAG4 persistence
corr_lag4 = analyze_firm_correlations(df_long, var1='RRR_PCT', var2='RRR_PCT_LAG4')

# Can also analyze acquisition rate persistence
corr_acq = analyze_firm_correlations(df_long, var1='ACQ_RATE_PCT', var2='RRR_PCT')

# Can also analyze acquisition rate persistence
corr_acq = analyze_firm_correlations(df_long, var1='RRR_PCT_LAG1', var2='PM_OPER')

# Can also analyze acquisition rate persistence
corr_acq = analyze_firm_correlations(df_long, var1='#GROWTH_MIX', var2='EXCESS_RET')

### function to run regressions and analysis 
### function to run regressions and analysis
### function to run regressions and analysis
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from linearmodels.panel import PanelOLS
from stargazer.stargazer import Stargazer

def analyze_columns(columns, firm_effect=True, time_effect=True, show_plots=True):
    # Calculate the correlation matrix among the selected columns
    corr_matrix = df_long[columns].corr()
    print("Correlation Matrix:")
    print(corr_matrix)
    
    # Compute summary statistics for each column
    print("\nSummary Statistics of columns:")
    summary_dict = {}
    for col in columns:
        col_data = df_long[col]
        summary_dict[col] = {
            'min': col_data.min(),
            'max': col_data.max(),
            'mean': col_data.mean(),
            'std' : col_data.std(),
            '25% quantile': col_data.quantile(0.25),
            '50% quantile': col_data.quantile(0.50),
            '75% quantile': col_data.quantile(0.75)
        }
    summary_df = pd.DataFrame(summary_dict).T
    print(summary_df)
    
    # Plot histograms (optional)
    if show_plots:
        for col in columns:
            lower = df_long[col].quantile(0.05) #winsorize for better display
            upper = df_long[col].quantile(0.95)
            data_filtered = df_long[col][(df_long[col] >= lower) & (df_long[col] <= upper)]
            plt.figure()
            plt.xlabel(col)
            plt.ylabel('Frequency')
            plt.hist(data_filtered)
            plt.title(f'Histogram of {col}')
            plt.show()
            
    # Run panel regression using PanelOLS
    if len(columns) > 1:
        dep_var = columns[0]
        indep_vars = columns[1:]
        y_reg = df_long[dep_var]
        X_reg = df_long[indep_vars]
        X_reg = sm.add_constant(X_reg)  # add constant
        
        model = PanelOLS(y_reg, X_reg, 
                         entity_effects=firm_effect, 
                         time_effects=time_effect,
                         drop_absorbed=True)
        results = model.fit(cov_type='clustered', cluster_entity=True, cluster_time=True)
        print("\nPanel Regression Results:")
        print(results.summary)
    else:
        print("Not enough columns provided for regression (need at least one independent variable).")




### Regression
### Regression
### Regression

analyze_columns(['RRR_PCT', 'RRR_PCT_LAG1','RRR_PCT_LAG2','RRR_PCT_LAG3','RRR_PCT_LAG4'], firm_effect=False, time_effect=False,show_plots=False)
#nothing
analyze_columns(['RRR_PCT', 'RRR_PCT_LAG4'], firm_effect=False, time_effect=False,show_plots=False)
#nothing
analyze_columns(['IS_SGA_EXPENSE_PCT','ACQ_RATE_PCT', 'ACQ_RATE_PCT_LAG1','RRR_PCT', 'RRR_PCT_LAG1'], firm_effect=True, time_effect=True,show_plots=False)
#Acq rate + RRR 
analyze_columns(['IS_SGA_EXPENSE', 'RETAINED_REV_EST', 'NEW_REV_EST'], firm_effect=True, time_effect=True,show_plots=False)
# ret revenue is significant but not new rev

analyze_columns(['EPR', 'RRR_PCT','RRR_PCT_LAG1'], firm_effect=True, time_effect=True,show_plots=False)

'''
multiply CLV by average profit margin
'''

analyze_columns(['IS_OPER_INC_GROWTH_PCT', 'ACQ_RATE_PCT', 'ACQ_RATE_PCT_LAG1','RET_REV_ACQ_LAG', 'RRR_PCT_LAG1'], firm_effect=True, time_effect=True,show_plots=False)
#nothing

analyze_columns(['IS_OPER_INC', 'RETAINED_REV_EST', 'NEW_REV_EST', 'IS_RD_EXPEND', 'IS_OTHER_OPER_INC', 'IS_COGS_TO_FE_AND_PP_AND_G','IS_SGA_EXPENSE'], firm_effect=True, time_effect=True,show_plots=False)
#nice
analyze_columns(['IS_OPER_INC', 'RET_REV_ACQ_LAG', 'RRR_PCT','IS_RD_EXPEND', 'IS_OTHER_OPER_INC', 'IS_COGS_TO_FE_AND_PP_AND_G','IS_SGA_EXPENSE'], firm_effect=True, time_effect=True,show_plots=False)
#nice
analyze_columns(['IS_OPER_INC','RET_REV_ACQ_LAG', 'RET_REV_ACQ_LAG2','RET_REV_ACQ_LAG3','RET_REV_ACQ_LAG4', 'IS_RD_EXPEND', 'IS_OTHER_OPER_INC', 'IS_COGS_TO_FE_AND_PP_AND_G','IS_SGA_EXPENSE'], firm_effect=True, time_effect=True,show_plots=False)
#nice

analyze_columns(['IS_OPER_INC','#GROWTH_MIX', 'RET_REV_ACQ_LAG', 'RET_REV_ACQ_LAG2','RET_REV_ACQ_LAG3','RET_REV_ACQ_LAG4', 'IS_RD_EXPEND', 'IS_OTHER_OPER_INC', 'IS_COGS_TO_FE_AND_PP_AND_G','IS_SGA_EXPENSE'], firm_effect=True, time_effect=True,show_plots=False)
#nice GM  not significant

analyze_columns(['PM_OPER', 'ACQ_RATE_PCT', 'ACQ_RATE_PCT_LAG1','RRR_PCT', 'RRR_PCT_LAG1'], firm_effect=True, time_effect=True,show_plots=False)
analyze_columns(['IS_OPER_INC_GROWTH_PCT','RRR_PCT', 'RRR_PCT_LAG1'], firm_effect=True, time_effect=True,show_plots=False)


'''
maybe add change of RRR -> degative trend signals future negative  growth
'''








'''
new rev est is not significant check why
'''

analyze_columns(['PM_OPER_PCT', 'RRR_PCT', 'IS_SGA_EXPENSE_PCT'], firm_effect=True, time_effect=True,show_plots=False)
analyze_columns(['PM_OPER_PCT', 'RRR_PCT', 'ACQ_RATE_PCT', 'IS_SGA_EXPENSE_PCT'], firm_effect=True, time_effect=True,show_plots=False)

analyze_columns(['IS_OPER_INC', 'RETAINED_REV_EST', 'NEW_REV_EST', 'IS_RD_EXPEND'], firm_effect=True, time_effect=True,show_plots=False)
analyze_columns(['IS_OPER_INC', 'RETAINED_REV_EST', 'NEW_REV_EST', 'IS_RD_EXPEND', 'IS_OTHER_OPER_INC', 'IS_COGS_TO_FE_AND_PP_AND_G'], firm_effect=True, time_effect=True,show_plots=True)
analyze_columns(['IS_OPER_INC', 'RETAINED_REV_EST', 'NEW_REV_EST'], firm_effect=True, time_effect=True,show_plots=False)
analyze_columns(['IS_OPER_INC', 'RETAINED_REV_EST', 'NEW_REV_EST'], firm_effect=False, time_effect=False,show_plots=False)



#predict
analyze_columns(['EXCESS_RET', 'MKT_RF_LAG', 'SIZE_LAG', 'BTM_LAG'], firm_effect=False, time_effect=False, show_plots=False)
analyze_columns(['EXCESS_RET', 'MKT_RF_LAG', 'SIZE_LAG', 'BTM_LAG','RRR_LAG'], firm_effect=False, time_effect=False, show_plots=False)
'''
to predict or to explain?
'''
#explain
analyze_columns(['EXCESS_RET', 'MKT_RF', 'SIZE', 'BTM','#RRR'], firm_effect=False, time_effect=False, show_plots=False)
analyze_columns(['EXCESS_RET', 'MKT_RF', 'SIZE', 'BTM','RRR_LAG'], firm_effect=True, time_effect=False, show_plots=False)

'''
why do I get those results?
- RRR is not significant, but the lagged RRR is significant
'''


#Lasso regression for variable selection
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# --- Step 1: Define variables ---
y_var = 'EXCESS_RET'
x_vars = [
    'MKT_RF','SIZE','BTM', 'RRR_PCT_LAG1','RRR_PCT_LAG2','IS_OPER_INC','IS_OPER_INC_LAG1',
    'RET_REV_ACQ_LAG','RET_REV_ACQ_LAG2','RET_REV_ACQ_LAG3','RET_REV_ACQ_LAG4',
    'RRR_PCT_LAG4','RRR_PCT','ACQ_RATE_PCT',#'ACQ_RATE_PCT_LAG1',#
    'ACQ_RATE_PCT_LAG4'
]

# --- Step 2: Prepare data ---
df_panel = df_long.reset_index()[['FIRM', 'DATE', y_var] + x_vars].dropna().copy()

# --- Step 3: Demean by firm (remove firm fixed effectsby commenting out) ---
#for col in [y_var] + x_vars:
#    df_panel[col] = df_panel.groupby('FIRM')[col].transform(lambda x: x - x.mean())

# --- Step 4: Define X and y ---
X = df_panel[x_vars].values
y = df_panel[y_var].values

# --- Step 5: Build LassoCV model ---
lasso_pipeline = Pipeline([
    ('scaler', StandardScaler()),   # normalize predictors
    ('lasso', LassoCV(cv=5, random_state=42, n_alphas=100, max_iter=20000))
])

# --- Step 6: Fit model ---
lasso_pipeline.fit(X, y)

lasso = lasso_pipeline.named_steps['lasso']

# --- Step 7: Display results ---
print("✅ Fixed-Effects Lasso Regression (Firm Demeaned Data)")
print(f"Optimal alpha (λ): {lasso.alpha_:.6f}")
print(f"R² (training): {lasso_pipeline.score(X, y):.4f}\n")

coef_df = pd.DataFrame({
    'Variable': x_vars,
    'Coefficient': lasso.coef_
})
coef_df['AbsCoef'] = coef_df['Coefficient'].abs()
coef_df = coef_df.sort_values('AbsCoef', ascending=False)

print("Top predictors by absolute coefficient:")
print(coef_df[['Variable', 'Coefficient']].head(10))

for col in ['IS_OPER_INC', 'EXCESS_RET']:
    low, high = df_long[col].quantile([0.01, 0.99])
    df_long[col] = df_long[col].clip(lower=low, upper=high)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(df_panel[x_vars])
std_effects = lasso.coef_ * scaler.scale_ / np.std(y)
print(std_effects)

#### XGBoost
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Step 1: Define y and X variables ---
y_var = 'EXCESS_RET'
x_vars = [
    'MKT_RF','SIZE','BTM','RRR_PCT_LAG1','RRR_PCT_LAG2','IS_OPER_INC','IS_OPER_INC_LAG1',
    'RET_REV_ACQ_LAG','RET_REV_ACQ_LAG2','RET_REV_ACQ_LAG3','RET_REV_ACQ_LAG4',
    'RRR_PCT_LAG4','RRR_PCT','ACQ_RATE_PCT','ACQ_RATE_PCT_LAG4'
]

# --- Step 2: Prepare data (drop missing) ---
df_model = df_long.reset_index()[['FIRM', 'DATE', y_var] + x_vars].dropna().copy()

# Optional: Clip extreme values
for col in ['IS_OPER_INC', 'EXCESS_RET']:
    low, high = df_model[col].quantile([0.01, 0.99])
    df_model[col] = df_model[col].clip(lower=low, upper=high)

# --- Step 3: Define X and y ---
X = df_model[x_vars].values
y = df_model[y_var].values

# --- Step 4: Split train/test (or use full data for in-sample performance) ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Step 5: Scale features (recommended for financial variables) ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Step 6: Fit XGBoost regressor ---
xgb = XGBRegressor(
    objective='reg:squarederror',
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
xgb.fit(X_train_scaled, y_train)

# --- Step 7: Evaluate performance ---
y_pred_train = xgb.predict(X_train_scaled)
y_pred_test = xgb.predict(X_test_scaled)

print("✅ XGBoost Model Performance")
print(f"R² (train): {r2_score(y_train, y_pred_train):.4f}")
print(f"R² (test):  {r2_score(y_test, y_pred_test):.4f}")

# --- Step 8: Feature importance ---
importances = xgb.feature_importances_
feat_df = pd.DataFrame({
    'Variable': x_vars,
    'Importance': importances
}).sort_values('Importance', ascending=False)

print("\nTop Predictors by Importance:")
print(feat_df.head(10))

# --- Step 9: Plot importance ---
plt.figure(figsize=(8, 5))
plt.barh(feat_df['Variable'], feat_df['Importance'])
plt.xlabel("Feature Importance")
plt.title("XGBoost Feature Importance")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

#explain features such as interactions

explainer = shap.Explainer(xgb)
shap_values = explainer(X_train_scaled)

shap.summary_plot(shap_values, X_train_scaled, feature_names=x_vars)

# Plot interaction between ACQ_RATE and RRR
#shap.plots.scatter(shap_values[:, "ACQ_RATE"], color=shap_values[:, "RRR"])

# Get SHAP interaction values (expensive computation!)
interaction_vals = shap.Explainer(xgb, X_train_scaled).shap_interaction_values(X_train_scaled)

# interaction_vals[i][j] gives interaction between variable i and j
# Sum over samples to get importance
interaction_importance = np.abs(interaction_vals).mean(axis=0)

# Create matrix of variable names
import pandas as pd
interaction_df = pd.DataFrame(interaction_importance, index=x_vars, columns=x_vars)

# Show top 10 strongest interaction pairs
interaction_df.where(np.triu(np.ones(interaction_df.shape), 1).astype(bool)).stack().sort_values(ascending=False).head(10)


#==============================================================================
# 7. Cash Flow Forecasting & Uncertainty Analysis (REVISED - SINGLE REGRESSION)
#==============================================================================
'''
summarize and clean up-> what do you really need?
'''


from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd

# --- Prepare data ---
X_vars = ['RRR_VOL', 'ACQ_VOL', 'RRR_LAST', 'ACQ_LAST']
y_var = 'CF_GROWTH_UNCERTAINTY'


# Drop missing rows
df_lasso = df_forecast_summary.dropna(subset=X_vars + [y_var]).copy()

# Define X and y
X = df_lasso[X_vars].values
y = df_lasso[y_var].values

# --- Build LassoCV pipeline ---
lasso_pipeline = Pipeline([
    ('scaler', StandardScaler()),   # standardize predictors
    ('lasso', LassoCV(cv=5, random_state=42, n_alphas=100, max_iter=10000))
])

# --- Fit model ---
lasso_pipeline.fit(X, y)

# Extract trained Lasso model
lasso = lasso_pipeline.named_steps['lasso']

# --- Results ---
print("✅ Lasso Regression Results")
print("----------------------------------")
print(f"Optimal alpha (λ): {lasso.alpha_:.6f}")
print(f"R² (training): {lasso.score(X, y):.4f}")
print()

# Coefficient summary
coef_df = pd.DataFrame({
    'Variable': X_vars,
    'Coefficient': lasso.coef_,
})
print(coef_df)

# Intercept
print(f"\nIntercept: {lasso.intercept_:.4f}")

# --- Identify which variables matter ---
important_vars = coef_df[coef_df['Coefficient'].abs() > 1e-6]
print("\nVariables retained by Lasso:")
print(important_vars if not important_vars.empty else "None (all shrunk to zero)")





# --- Helper: safe SARIMA forecast ---

# Add global counter dictionary at the top of your forecast section
SARIMA_STATS = {
    'total_forecasts': 0,
    'success': 0,
    'too_few_obs': 0,
    'model_failed': 0
}

from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
def safe_sarima_forecast(y_train):
    """
    Fit SARIMA safely and return (forecast_value, fallback_flag).
    fallback_flag: 0=success, 1=too_few_obs, 2=model_failed
    """

    global SARIMA_STATS
    SARIMA_STATS['total_forecasts'] += 1
    
    fallback_flag = 0
    y_pred = np.nan

    y_train = pd.to_numeric(y_train, errors='coerce').dropna()

    if len(y_train) < 6 or np.nanvar(y_train) < 1e-8:
        fallback_flag = 1
        SARIMA_STATS['too_few_obs'] += 1
        return (y_train.iloc[-1] if len(y_train) > 0 else np.nan, fallback_flag)

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            
            model = SARIMAX(
                y_train,
                order=(1, 1, 1),
                seasonal_order=(1, 0, 1, 4),
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            res = model.fit(disp=False, maxiter=50)
            forecast = res.forecast(steps=1)
            
            y_pred = float(forecast.iloc[0]) if hasattr(forecast, 'iloc') else float(forecast)
            SARIMA_STATS['success'] += 1
            
    except Exception as e:
        fallback_flag = 2
        SARIMA_STATS['model_failed'] += 1
        y_pred = float(y_train.iloc[-1]) if len(y_train) > 0 else np.nan

    return y_pred, fallback_flag


# --- Helper: last valid split ---
def last_valid_split(series, dates=None):
    """
    Split into train/test, returning SCALAR test value (not Series).
    Returns (y_train_series, y_test_scalar) or (None, None).
    """
    s = pd.to_numeric(series, errors='coerce').dropna()
    
    if len(s) < 6:
        return None, None
    
    # If dates provided, align
    if dates is not None and len(dates) == len(series):
        valid_dates = pd.to_datetime(dates[s.index])
        s.index = valid_dates
    
    # Return: (training series, scalar test value)
    return s.iloc[:-1], float(s.iloc[-1])


# --- Main Function ---

'''
summarize firm_forecast_summary & summary_uncertainty
'''

def firm_forecast_summary(group):
    group = group.sort_values('DATE').copy()

    # Ensure numeric columns
    y_inc = pd.to_numeric(group['IS_OPER_INC'], errors='coerce')
    y_pm  = pd.to_numeric(group['PM_OPER'], errors='coerce')
    y_rg  = pd.to_numeric(group.get('#REVENUE_GROWTH', np.nan), errors='coerce')
    y_cfg = pd.to_numeric(group.get('IS_OPER_INC_GROWTH', np.nan), errors='coerce')
    rrr   = pd.to_numeric(group['RRR_PCT'], errors='coerce')
    acq   = pd.to_numeric(group['ACQ_RATE_PCT'], errors='coerce')

    # --- Safe splits ---
    splits = {var: last_valid_split(series) for var, series in {
        'CF': y_inc, 'PM': y_pm, 'RG': y_rg, 'CFG': y_cfg
    }.items()}

    # Initialize forecast and uncertainty containers
    forecasts = {}
    uncertainties = {}

    for key, (y_train, y_test) in splits.items():
        if y_train is None or y_test is None:
            forecasts[key] = np.nan
            uncertainties[key] = np.nan
        else:
            y_pred, _ = safe_sarima_forecast(y_train)
            forecasts[key] = y_pred


            # ABSOLUTE PERCENTAGE ERROR (APE)
            # Formula: |Actual - Forecast| / |Actual| * 100
            if y_test != 0:
                uncertainties[key] = (abs(y_test - y_pred) / abs(y_test)) * 100
            else:
                uncertainties[key] = np.nan  # Can't divide by zero

    # --- RRR / Acquisition metrics from training period ---
    rrr_clean = rrr.dropna()
    acq_clean = acq.dropna()

    rrr_last = rrr_clean.iloc[-2] if len(rrr_clean) >= 2 else np.nan
    acq_last = acq_clean.iloc[-2] if len(acq_clean) >= 2 else np.nan
    rrr_vol  = np.nanstd(rrr_clean)
    acq_vol  = np.nanstd(acq_clean)

    # --- Volatilities ---
    rg_vol   = np.nanstd(splits['RG'][0]) if splits['RG'][0] is not None else np.nan
    cfg_vol  = np.nanstd(splits['CFG'][0]) if splits['CFG'][0] is not None else np.nan
    pm_vol   = np.nanstd(splits['PM'][0]) if splits['PM'][0] is not None else np.nan

    # --- Return structured result ---
    result = {
        'FIRM': group['FIRM'].iloc[0],
        'CF_FORECAST_LAST': forecasts['CF'],
        'PM_FORECAST_LAST': forecasts['PM'],
        'RG_FORECAST_LAST': forecasts['RG'],
        'CF_GROWTH_FORECAST_LAST': forecasts['CFG'],
        'CF_UNCERTAINTY': uncertainties['CF'],
        'PM_UNCERTAINTY': uncertainties['PM'],
        'RG_UNCERTAINTY': uncertainties['RG'],
        'CF_GROWTH_UNCERTAINTY': uncertainties['CFG'],
        'RRR_LAST': rrr_last,
        'ACQ_LAST': acq_last,
        'RRR_VOL': rrr_vol,
        'ACQ_VOL': acq_vol,
        'RG_VOL': rg_vol,
        'CF_GROWTH_VOL': cfg_vol,
        'PM_VOL': pm_vol
    }

    return pd.Series(result, name=group['FIRM'].iloc[0])


print("\n=== Generating forecast summary (this may take a few minutes) ===")
print(f"Processing {df_long.index.get_level_values('FIRM').nunique()} firms...")

# Apply the function to each firm
df_forecast_summary = (
    df_long.reset_index()
    .groupby('FIRM', group_keys=False)
    .apply(firm_forecast_summary)
    .reset_index(drop=True)
)

print("✅ Forecast summary created successfully.")
print(f"\nShape: {df_forecast_summary.shape}")
print(f"Columns: {list(df_forecast_summary.columns)}")
print("\nFirst few rows:")
print(df_forecast_summary.head(5))


# --- Print Quartiles for Each Metric ---
# Select the uncertainty columns
from scipy.stats.mstats import winsorize
uncertainty_cols = ['PM_UNCERTAINTY', 'RG_UNCERTAINTY', 'CF_GROWTH_UNCERTAINTY']
# Prepare data for plotting (melt to long format)
df_plot = df_forecast_summary[uncertainty_cols].melt(
    var_name='Uncertainty_Type', 
    value_name='APE_Percentage'
)

# Remove NaN values
df_plot = df_plot.dropna()

# --- Summary Statistics per Uncertainty Type ---
print("\n=== Summary Statistics by Uncertainty Type ===")
summary_by_type = df_plot.groupby('Uncertainty_Type')['APE_Percentage'].describe()
print(summary_by_type.round(2))

# --- Boxplot WITH Winsorization (FIXED) ---
print("\n=== Creating Boxplot (Winsorized for Display) ===")

# Create a COPY and drop NaNs FIRST, then winsorize
df_plot_winsorized = pd.DataFrame()

for col in uncertainty_cols:
    # Drop NaNs for this column
    clean_data = df_forecast_summary[col].dropna()
    
    if len(clean_data) > 0:
        # Winsorize the clean data
        winsorized_values = winsorize(clean_data.values, limits=[0.00, 0.1])
        
        # Create a temporary dataframe with the winsorized values
        temp_df = pd.DataFrame({
            'Uncertainty_Type': col,
            'APE_Percentage': winsorized_values
        })
        
        df_plot_winsorized = pd.concat([df_plot_winsorized, temp_df], ignore_index=True)

# Create winsorized boxplot
plt.figure(figsize=(12, 6))
sns.boxplot(data=df_plot_winsorized, x='Uncertainty_Type', y='APE_Percentage', palette='Set3')

plt.title('Distribution of Forecast Errors (Winsorized at 90%)', fontsize=14, fontweight='bold')
plt.xlabel('Forecast Uncertainty Metric', fontsize=12)
plt.ylabel('Absolute Percentage Error (%) - Winsorized', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()


# Check for any issues
print(f"\n=== Data Quality Check ===")
print(f"Total firms: {len(df_forecast_summary)}")
print(f"Firms with valid CF_UNCERTAINTY: {df_forecast_summary['CF_UNCERTAINTY'].notna().sum()}")
print(f"Firms with valid RRR_VOL: {df_forecast_summary['RRR_VOL'].notna().sum()}")





# --- Distribution summary for uncertainty measures ---

summary_uncertainty = pd.DataFrame({
    'Variable': ['CF_UNCERTAINTY', 'PM_UNCERTAINTY', 'RG_UNCERTAINTY', 'CF_GROWTH_UNCERTAINTY', 'RRR_VOL', 'ACQ_VOL', 'RG_VOL'],
    'Mean': [
        df_forecast_summary['CF_UNCERTAINTY'].mean(skipna=True),
        df_forecast_summary['PM_UNCERTAINTY'].mean(skipna=True),
        df_forecast_summary['RG_UNCERTAINTY'].mean(skipna=True),
        df_forecast_summary['CF_GROWTH_UNCERTAINTY'].mean(skipna=True),
        df_forecast_summary['RRR_VOL'].mean(skipna=True),
        df_forecast_summary['ACQ_VOL'].mean(skipna=True),
        df_forecast_summary['RG_VOL'].mean(skipna=True)
    ],
    'Median': [
        df_forecast_summary['CF_UNCERTAINTY'].median(skipna=True),
        df_forecast_summary['PM_UNCERTAINTY'].median(skipna=True),
        df_forecast_summary['RG_UNCERTAINTY'].median(skipna=True),
        df_forecast_summary['CF_GROWTH_UNCERTAINTY'].median(skipna=True),
        df_forecast_summary['RRR_VOL'].median(skipna=True),
        df_forecast_summary['ACQ_VOL'].median(skipna=True),
        df_forecast_summary['RG_VOL'].median(skipna=True)
    ],
    'StdDev': [
        df_forecast_summary['CF_UNCERTAINTY'].std(skipna=True),
        df_forecast_summary['PM_UNCERTAINTY'].std(skipna=True),
        df_forecast_summary['RG_UNCERTAINTY'].std(skipna=True),
        df_forecast_summary['CF_GROWTH_UNCERTAINTY'].std(skipna=True),
        df_forecast_summary['RRR_VOL'].std(skipna=True),
        df_forecast_summary['ACQ_VOL'].std(skipna=True),
        df_forecast_summary['RG_VOL'].std(skipna=True)
    ],
    'Min': [
        df_forecast_summary['CF_UNCERTAINTY'].min(skipna=True),
        df_forecast_summary['PM_UNCERTAINTY'].min(skipna=True),
        df_forecast_summary['RG_UNCERTAINTY'].min(skipna=True),
        df_forecast_summary['CF_GROWTH_UNCERTAINTY'].min(skipna=True),
        df_forecast_summary['RRR_VOL'].min(skipna=True),
        df_forecast_summary['ACQ_VOL'].min(skipna=True),
        df_forecast_summary['RG_VOL'].min(skipna=True)
    ],
    'Max': [
        df_forecast_summary['CF_UNCERTAINTY'].max(skipna=True),
        df_forecast_summary['PM_UNCERTAINTY'].max(skipna=True),
        df_forecast_summary['RG_UNCERTAINTY'].max(skipna=True),
        df_forecast_summary['CF_GROWTH_UNCERTAINTY'].max(skipna=True),
        df_forecast_summary['RRR_VOL'].max(skipna=True),
        df_forecast_summary['ACQ_VOL'].max(skipna=True),
        df_forecast_summary['RG_VOL'].max(skipna=True)
    ]
})

print("📊 Forecast Uncertainty Summary:")
print(summary_uncertainty.round(2))


uncertainty_cols = ['CF_UNCERTAINTY', 'PM_UNCERTAINTY', 'RG_UNCERTAINTY', 'CF_GROWTH_UNCERTAINTY']

na_summary = (
    df_forecast_summary[uncertainty_cols]
    .isna()
    .sum()
    .reset_index()
    .rename(columns={'index': 'Variable', 0: 'NaN_Count'})
)

# Also show % of missing values
na_summary['NaN_%'] = (
    na_summary['NaN_Count'] / len(df_forecast_summary) * 100
).round(2)

print("📊 Missing Value Summary (per Uncertainty Variable):")
print(na_summary)



import statsmodels.formula.api as smf

# Drop rows with missing relevant data
df_reg = df_forecast_summary.dropna(
    subset=['RG_UNCERTAINTY', 'RRR_VOL', 'RRR_LAST']
).copy()

# Optionally winsorize extreme uncertainty values for stability
from scipy.stats.mstats import winsorize
df_reg['RG_UNCERTAINTY_W'] = winsorize(df_reg['RG_UNCERTAINTY'], limits=[0.01, 0.01])

# --- Run regression ---
model = smf.ols(
    formula="RG_UNCERTAINTY_W ~ RRR_VOL + RRR_LAST",
    data=df_reg
).fit(cov_type='HC3')

print(model.summary())


'''
new plan: repeat steps but with revenue data; new and retained revenue estimates
'''
def firm_forecast_summary_customers(group):
    """
    Forecast customers & customer metrics per firm.
    Returns a Series (NOT DataFrame) with scalar values.
    """
    group = group.sort_values('DATE').copy()
    
    # Get firm name once
    firm_name = group['FIRM'].iloc[0]
    
    # Dates (for potential alignment, though we're now returning scalars)
    dates = pd.to_datetime(group['DATE'])

    # Ensure numeric series
    y_new = pd.to_numeric(group['#NEW_CUSTOMERS'], errors='coerce')
    y_ret = pd.to_numeric(group['#RETURNING_CUSTOMERS'], errors='coerce')
    acq   = pd.to_numeric(group['ACQ_RATE_PCT'], errors='coerce')
    rrr   = pd.to_numeric(group['RRR_PCT'], errors='coerce')

    # --- Safe splits (now returns scalar test values) ---
    splits = {
        'NEW': last_valid_split(y_new, dates),
        'RET': last_valid_split(y_ret, dates),
        'ACQ': last_valid_split(acq, dates),
        'RRR': last_valid_split(rrr, dates)
    }

    # --- Forecasts & uncertainties (all scalars now) ---
    forecasts = {}
    uncertainties = {}

    for key, (y_train, y_test) in splits.items():
        if y_train is None or y_test is None or pd.isna(y_test):
            forecasts[key] = np.nan
            uncertainties[key] = np.nan
        else:
            y_pred, _ = safe_sarima_forecast(y_train)  # returns scalar
            forecasts[key] = y_pred

            # ABSOLUTE PERCENTAGE ERROR
            if y_test != 0:
                uncertainties[key] = (abs(y_test - y_pred) / abs(y_test)) * 100
            else:
                uncertainties[key] = np.nan     

    # --- Volatilities (scalars) ---
    acq_vol = float(np.nanstd(acq)) if len(acq.dropna()) > 1 else np.nan
    rrr_vol = float(np.nanstd(rrr)) if len(rrr.dropna()) > 1 else np.nan

    # --- Last values (scalars) ---
    acq_last = float(acq.iloc[-2]) if len(acq.dropna()) >= 2 else np.nan
    rrr_last = float(rrr.iloc[-2]) if len(rrr.dropna()) >= 2 else np.nan

    # --- Return as Series with SCALAR values ---
    result = pd.Series({
        'FIRM': firm_name,
        'NEW_FORECAST_LAST': forecasts['NEW'],
        'RET_FORECAST_LAST': forecasts['RET'],
        'ACQ_FORECAST_LAST': forecasts['ACQ'],
        'RRR_FORECAST_LAST': forecasts['RRR'],
        'NEW_UNCERTAINTY': uncertainties['NEW'],
        'RET_UNCERTAINTY': uncertainties['RET'],
        'ACQ_UNCERTAINTY': uncertainties['ACQ'],
        'RRR_UNCERTAINTY': uncertainties['RRR'],
        'ACQ_VOL': acq_vol,
        'RRR_VOL': rrr_vol,
        'ACQ_LAST': acq_last,
        'RRR_LAST': rrr_last
    }, name=firm_name)

    return result


# --- Apply across firms ---
df_forecast_summary = (
    df_long.reset_index()
    .groupby('FIRM', group_keys=False)
    .apply(firm_forecast_summary_customers)
    .reset_index(drop=True)
)

print("✅ Forecast summary created successfully.")
print(f"\nShape: {df_forecast_summary.shape}")
print("\nFirst few rows:")
print(df_forecast_summary.head(5))


# After running the forecasts, print summary report
print("\n" + "="*60)
print("SARIMA FORECASTING SUMMARY REPORT")
print("="*60)

total = SARIMA_STATS['total_forecasts']
success = SARIMA_STATS['success']
too_few = SARIMA_STATS['too_few_obs']
failed = SARIMA_STATS['model_failed']

print(f"\nTotal forecast attempts: {total}")
print(f"Successful SARIMA fits: {success} ({success/total*100:.1f}%)")
print(f"Too few observations: {too_few} ({too_few/total*100:.1f}%)")
print(f"Model convergence failures: {failed} ({failed/total*100:.1f}%)")
print(f"\nFallback to last value: {too_few + failed} ({(too_few + failed)/total*100:.1f}%)")

# Check for missing values
print("\n=== Missing Value Summary ===")
missing_summary = df_forecast_summary.isna().sum()
print(missing_summary[missing_summary > 0])

# --- Summary stats ---
summary_cols = ['NEW_UNCERTAINTY', 'RET_UNCERTAINTY', 'ACQ_UNCERTAINTY', 
                'RRR_UNCERTAINTY', 'ACQ_VOL', 'RRR_VOL']
summary_uncertainty = df_forecast_summary[summary_cols].describe().T

print("\n📊 Forecast Uncertainty and Volatility Summary:")
print(summary_uncertainty.round(4))

#==============================================================================
# 6. Portfolio Analysis
#==============================================================================

'''
neu anfang
'''
file3_path = "C:\\Users\\thkraft\\eCommerce-Goethe Dropbox\\Thilo Kraft\\Thilo(privat)\\Privat\\Research\\RRR_FinancialImplication\\Data\\2025-07-02a-TK-Monthly-Returns_Python.xlsx"
try:
    # Read the file
    monthly_returns = pd.read_excel(file3_path, header=None)
    print("Monthly Return data sucessfully read in")
    print("Shape:", monthly_returns.shape)

except Exception as e:
    #Exception is the base class for all exceptions (FilenotFoundError, ValueError, etc.)
    #As e saves the "exception"
    print(f"Fehler beim Einlesen der Dateien: {e}")

# Step 1: Process Headers
header_rows_monthly = monthly_returns.iloc[:2];  # First two rows as headers
data_rows_monthly = monthly_returns.iloc[2:];  # Remaining rows are data


# Combine headers: "Firm.Variable"
combined_headers_monthly = header_rows_monthly.apply(lambda x: x.astype(str).str.strip(), axis=0);
column_headers_monthly = combined_headers_monthly.apply(lambda x: '.'.join(x.dropna()), axis=0);
data_rows_monthly.columns = column_headers_monthly;  # Assign proper headers to the data rows
df_monthly_returns = data_rows_monthly.reset_index(drop=True);  # Reset index with proper columns


# Rename the first column to "Date"
data_rows_monthly.rename(columns={data_rows_monthly.columns[0]: 'Date'}, inplace=True);

# Reset the index for the data rows
df_monthly_returns = data_rows_monthly.reset_index(drop=True);
df_monthly_returns.head()
df_monthly_returns['Date'] = pd.to_datetime(df_monthly_returns['Date'], errors='coerce')  # Convert 'Date' to datetime
df_monthly_returns.head()

# Step 2: Handle Missing Data
df_monthly_returns.replace('#N/A N/A', np.nan, inplace=True)


#reshape wide to long format

df_prices = df_monthly_returns.copy()

# Make sure all columns except 'Date' are numeric
for col in df_prices.columns:
    if col != 'Date':
        df_prices[col] = pd.to_numeric(df_prices[col], errors='coerce')

# Ensure Date is datetime
df_prices['Date'] = pd.to_datetime(df_prices['Date'], format='%d/%m/%Y')

# Calculate log returns (excluding 'Date')
px_only = df_prices.drop(columns=['Date'])
returns_px = np.log(px_only / px_only.shift(1))

# Add Date column back in
returns_px['Date'] = df_prices['Date'].values

# Melt to long format
returns = returns_px.melt(id_vars='Date', var_name='FirmVar', value_name='RETURN_LOG')
returns = returns.dropna(subset=['RETURN_LOG'])
returns['FIRM'] = returns['FirmVar'].str.replace('.PX_LAST', '', regex=False)
returns = returns.drop(columns=['FirmVar'])

print(returns.head())


# This ensures Jan-Mar 2020 use Q4 2019 RRR_LAG (which is Q3 2019 RRR)
### Alternativly use ty the 1 month shift for reporting dates
returns['QUARTER'] = (
    returns['Date']
    .dt.to_period('Q')
    .apply(lambda x: x - 1)  # Shift back one quarter
    .dt.to_timestamp('Q', 'end')
)
'''
# Assign each month to its quarter end
returns['QUARTER'] = returns['Date'].dt.to_period('Q').dt.to_timestamp('Q', 'end')
it should assing each quarter to its quarter start

'''



# Prepare the quarterly RRR data
df_quarterly = (
    df_long.reset_index()[['FIRM', 'DATE', 'RRR_LAG', 'HISTORICAL_MARKET_CAP']]
    .copy()
)
df_quarterly['DATE'] = pd.to_datetime(df_quarterly['DATE'])
# Snap to quarter-end (in case it's not)
df_quarterly['QUARTER'] = df_quarterly['DATE'].dt.to_period('Q').dt.to_timestamp('Q', 'end')

#Step 3: Merge Quarterly RRR Signal onto Monthly Returns


#Prepare the Quarterly RRR Data
# For monthly returns:
returns['FIRM'] = returns['FIRM'].str.upper()
# Get set of firms from both sources
firms_monthly = set(returns['FIRM'].unique())
firms_quarterly = set(df_long.reset_index()['FIRM'].unique())

# Firms in monthly but not in quarterly
missing_in_quarterly = firms_monthly - firms_quarterly
print(f"Firms in monthly data but not in quarterly data: {missing_in_quarterly}")

# Firms in quarterly but not in monthly
missing_in_monthly = firms_quarterly - firms_monthly
print(f"Firms in quarterly data but not in monthly data: {missing_in_monthly}")

# Intersection (firms present in both)
common_firms = firms_monthly & firms_quarterly
print(f"Number of common firms: {len(common_firms)}")

returns = returns.merge(
    df_quarterly[['FIRM', 'QUARTER', 'RRR_LAG', 'HISTORICAL_MARKET_CAP']],
    on=['FIRM', 'QUARTER'],
    how='left'
)


#Assign RRR quartiles by quarter
returns['RRR_LAG'] = pd.to_numeric(returns['RRR_LAG'], errors='coerce')

def safe_qcut(x):
    x = x.dropna()
    if x.nunique() < 4:
        return pd.Series([np.nan]*len(x), index=x.index)
    return pd.qcut(x, 4, labels=['Q1','Q2','Q3','Q4'])

returns['quartile'] = (
    returns.groupby('QUARTER')['RRR_LAG']
    .transform(safe_qcut)
)
#first quarter has no RRR_LAG, so quartile is NaN
#Use safe_qcut to handle cases with fewer than 4 unique values


### Build the portfolios
def calc_and_plot_portfolio_returns_from_long(
    returns,
    weight_type='value',
    benchmarks=['SPX INDEX', 'SPW INDEX']
):
    """
    Calculate and plot monthly portfolio returns by quartile, plus included benchmarks from returns DataFrame.
    Args:
        returns: DataFrame with columns ['Date', 'quartile', 'RETURN_LOG', 'FIRM', 'HISTORICAL_MARKET_CAP']
        weight_type: 'equal' or 'value' (default 'value')
        benchmarks: list of FIRM names to use as benchmarks (must exist in returns['FIRM'])
    Returns:
        port_rets: DataFrame of monthly log returns (portfolios + benchmarks)
        cum_returns: DataFrame of cumulative returns
    """
    # Portfolio log returns
    if weight_type == 'equal':
        port_rets = (
            returns.groupby(['Date', 'quartile'])['RETURN_LOG']
            .mean()
            .unstack('quartile')
            .sort_index()
        )
    else:  # value-weighted
        returns['MCAP'] = pd.to_numeric(returns['HISTORICAL_MARKET_CAP'], errors='coerce')
        returns['MCAP_SUM'] = returns.groupby(['Date', 'quartile'])['MCAP'].transform('sum')
        returns['w'] = returns['MCAP'] / returns['MCAP_SUM']
        returns['w_return'] = returns['w'] * returns['RETURN_LOG']
        port_rets = (
            returns.groupby(['Date', 'quartile'])['w_return']
            .sum()
            .unstack('quartile')
            .sort_index()
        )

    # Add benchmarks (from returns, not price)
    for bmk in benchmarks:
        if bmk in returns['FIRM'].unique():
            # Group by Date, mean in case there are duplicates (shouldn't be)
            ser = (
                returns[returns['FIRM'] == bmk]
                .groupby('Date')['RETURN_LOG']
                .mean()
                .reindex(port_rets.index)
            )
            port_rets[bmk.split()[0]] = ser.values

    # Cumulative returns (start at 0)
    cum_returns = np.exp(port_rets.cumsum()) - 1
    # Prepend a zero for plotting
    first_date = cum_returns.index.min()
    prior_date = first_date - pd.offsets.MonthEnd(1)
    start_row = pd.DataFrame({c: 0.0 for c in cum_returns.columns}, index=[prior_date])
    cum_returns = pd.concat([start_row, cum_returns]).sort_index()

    # Plot
    plt.figure(figsize=(12, 6))
    for col in cum_returns.columns:
        plt.plot(cum_returns.index, cum_returns[col], label=col)
    plt.title(f'Cumulative Returns by RRR Quartile Portfolio ({weight_type.capitalize()}-Weighted)')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return (Start = 0)')
    plt.legend(title='Portfolio', loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    return port_rets, cum_returns


port_rets_vw, cum_returns_vw = calc_and_plot_portfolio_returns_from_long(
    returns, weight_type='value', benchmarks=['SPX INDEX']
)

port_rets_eq, cum_returns_eq = calc_and_plot_portfolio_returns_from_long(
    returns, weight_type='equal', benchmarks=['SPW INDEX']
)

# Calculate value-weighted market return for all sample firms (excluding SPX)
sample_firms = returns[~returns['FIRM'].isin(['SPX INDEX', 'SPW INDEX'])].copy()
sample_firms['MCAP'] = pd.to_numeric(sample_firms['HISTORICAL_MARKET_CAP'], errors='coerce')

# Calculate total market cap per month
sample_firms['MCAP_TOTAL'] = sample_firms.groupby('Date')['MCAP'].transform('sum')
sample_firms['w'] = sample_firms['MCAP'] / sample_firms['MCAP_TOTAL']
sample_firms['w_return'] = sample_firms['w'] * sample_firms['RETURN_LOG']

# Aggregate to get market return
market_ret = (
    sample_firms.groupby('Date')['w_return']
    .sum()
    .sort_index()
)

# Add to portfolio returns
port_rets_vw['SAMPLE_VW'] = market_ret.reindex(port_rets_vw.index)

# Recalculate cumulative returns with new benchmark
cum_returns_vw_updated = np.exp(port_rets_vw.cumsum()) - 1
first_date = cum_returns_vw_updated.index.min()
prior_date = first_date - pd.offsets.MonthEnd(1)
start_row = pd.DataFrame({c: 0.0 for c in cum_returns_vw_updated.columns}, index=[prior_date])
cum_returns_vw_updated = pd.concat([start_row, cum_returns_vw_updated]).sort_index()

# Plot with updated benchmark
plt.figure(figsize=(12, 6))
for col in cum_returns_vw_updated.columns:
    linestyle = '--' if col in ['SPX', 'SAMPLE_VW'] else '-'
    linewidth = 2 if col == 'SAMPLE_VW' else 1.5
    plt.plot(cum_returns_vw_updated.index, cum_returns_vw_updated[col], 
             label=col, linestyle=linestyle, linewidth=linewidth)

plt.title('Cumulative Returns by RRR Quartile Portfolio (Value-Weighted)')
plt.xlabel('Date')
plt.ylabel('Cumulative Return (Start = 0)')
plt.legend(title='Portfolio', loc='upper left')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()



# Analysing Growth Mix by Quartile with Time Alignment

# =============================================================================
# Expanded Distribution Analysis by Quartile (RRR, Acquisition Rate, Growth Mix)
# =============================================================================

'''
technically I also need to use N-2 fundamentals
talk to someone about the timeframe alignment
'''

def metric_distribution_by_quartile(returns_df, metric_tag):
    """
    Analyze distribution of any metric across RRR quartiles.
    Uses time-aligned quartile assignments and metric values.
    
    Parameters:
    -----------
    returns_df : DataFrame with columns ['Date', 'FIRM', 'quartile']
    metric_tag : str, column name in df_long (e.g., '#GROWTH_MIX', 'RRR_PCT', 'ACQ_RATE_PCT')
    """
    # Prepare returns data - create QUARTER from Date
    returns_temp = returns_df[['Date', 'FIRM', 'quartile']].copy()
    returns_temp = returns_temp.dropna(subset=['quartile'])
    
    # Create QUARTER column from Date
    returns_temp['QUARTER'] = pd.to_datetime(returns_temp['Date']).dt.to_period('Q').dt.to_timestamp('Q', 'end')
    
    # Get metric from df_long with proper time alignment
    df_temp = df_long.reset_index()
    if metric_tag not in df_temp.columns:
        print(f"⚠️ Column '{metric_tag}' not found in df_long")
        return None
    
    df_temp[metric_tag] = pd.to_numeric(df_temp[metric_tag], errors='coerce')
    df_temp['QUARTER'] = pd.to_datetime(df_temp['DATE']).dt.to_period('Q').dt.to_timestamp('Q', 'end')
    
    # Merge on FIRM and QUARTER to align time periods
    merged = returns_temp.merge(
        df_temp[['FIRM', 'QUARTER', metric_tag]], 
        on=['FIRM', 'QUARTER'], 
        how='left'
    )
    
    # Check merge success
    print(f"Merged rows: {len(merged)}, Non-null {metric_tag}: {merged[metric_tag].notna().sum()}")
    
    # Calculate stats by quartile
    stats = (
        merged.groupby('quartile')[metric_tag]
        .agg(['mean', 'median', lambda x: x.quantile(0.25), lambda x: x.quantile(0.75), 'std', 'count'])
    )
    stats.columns = ['mean', 'median', 'Q25', 'Q75', 'std', 'count']
    
    return stats


print("\n" + "="*80)
print("DISTRIBUTION ANALYSIS BY RRR QUARTILE")
print("="*80)

# Growth Mix Statistics
print("\n--- Growth Mix Statistics by Quartile (Time-Aligned) ---")
gm_stats = metric_distribution_by_quartile(returns, '#GROWTH_MIX')
if gm_stats is not None:
    print(gm_stats.round(2))

# RRR Distribution Statistics
print("\n--- RRR Distribution by Quartile (Time-Aligned) ---")
rrr_stats = metric_distribution_by_quartile(returns, 'RRR_PCT')
if rrr_stats is not None:
    print(rrr_stats.round(2))

# Acquisition Rate Distribution Statistics
print("\n--- Acquisition Rate Distribution by Quartile (Time-Aligned) ---")
acq_stats = metric_distribution_by_quartile(returns, 'ACQ_RATE_PCT')
if acq_stats is not None:
    print(acq_stats.round(2))


# Revenue Growth Rate Distribution Statistics
print("\n--- Revenue Growth Rate Distribution by Quartile (Time-Aligned) ---")
rev_growth_stats = metric_distribution_by_quartile(returns, 'REV_GROWTH_PCT')
if rev_growth_stats is not None:
    print(rev_growth_stats.round(2))

# Size (Log Market Cap) Distribution Statistics
print("\n--- Size (Log Market Cap) Distribution by Quartile (Time-Aligned) ---")
size_stats = metric_distribution_by_quartile(returns, 'SIZE')
if size_stats is not None:
    print(size_stats.round(2))

# Book-to-Market (BTM) Distribution Statistics
print("\n--- Book-to-Market Ratio Distribution by Quartile (Time-Aligned) ---")
btm_stats = metric_distribution_by_quartile(returns, 'BTM')
if btm_stats is not None:
    print(btm_stats.round(2))

# Profit Margin Distribution Statistics
print("\n--- Operating Profit Margin Distribution by Quartile (Time-Aligned) ---")
pm_stats = metric_distribution_by_quartile(returns, 'PM_OPER_PCT')
if pm_stats is not None:
    print(pm_stats.round(2))

# Revenue (Sales) Distribution Statistics
print("\n--- Revenue (Sales) Distribution by Quartile (Time-Aligned) ---")
revenue_stats = metric_distribution_by_quartile(returns, 'SALES_REV_TURN')
if revenue_stats is not None:
    print(revenue_stats.round(2))

# Optional: Create a combined summary table with ALL metrics
print("\n--- Combined Summary: Median Values by Quartile ---")
combined_summary = pd.DataFrame({
    'RRR (%)': rrr_stats['median'] if rrr_stats is not None else np.nan,
    'Acq Rate (%)': acq_stats['median'] if acq_stats is not None else np.nan,
    'Growth Mix': gm_stats['median'] if gm_stats is not None else np.nan,
    'Rev Growth (%)': rev_growth_stats['median'] if rev_growth_stats is not None else np.nan,
    'Size (log)': size_stats['median'] if size_stats is not None else np.nan,
    'BTM': btm_stats['median'] if btm_stats is not None else np.nan,
    'PM (%)': pm_stats['median'] if pm_stats is not None else np.nan,
    'Revenue (M)': revenue_stats['median'] if revenue_stats is not None else np.nan
})
print(combined_summary.round(2))








import statsmodels.api as sm
from scipy.stats import skew, kurtosis
def portfolio_performance_table(port_ret, benchmark_cols=['SPX', 'SAMPLE_VW']):
    """
    Calculate performance metrics for ALL columns (including benchmarks).
    
    Parameters:
    -----------
    port_ret : DataFrame with portfolio returns
    benchmark_cols : list of benchmark column names (for reference, all get metrics)
    """
    measures = [
        'Geometric Mean Return', 'Downside Deviation', 'Max Drawdown',
        'Sortino Ratio', 'Skewness', 'Kurtosis', 
        'Alpha (vs SPX)', 'Beta (vs SPX)',
        'Alpha (vs SAMPLE_VW)', 'Beta (vs SAMPLE_VW)',
        'VaR 5%', 'Hit Ratio'
    ]
    
    # Include ALL columns (portfolios + benchmarks)
    all_columns = list(port_ret.columns)
    perf = pd.DataFrame(index=measures, columns=all_columns)

    for col in all_columns:
        returns = port_ret[col].dropna()
        
        # Geometric mean
        geo_mean = np.exp(returns.mean()) - 1

        # Downside deviation (negative returns only)
        downside = returns[returns < 0]
        dd = downside.std(ddof=0)

        # Max drawdown
        cum = np.exp(returns.cumsum())
        cum_max = cum.cummax()
        drawdown = (cum / cum_max - 1).min()

        # Sortino ratio
        sortino = returns.mean() / dd if dd > 0 else np.nan

        # Skewness and kurtosis
        skewness = skew(returns, nan_policy='omit')
        kurt = kurtosis(returns, nan_policy='omit', fisher=False)

        # Alpha and Beta for EACH benchmark
        # NOTE: Benchmarks will have Alpha/Beta vs themselves = 0/1 (or vs other benchmarks)
        alphas = {}
        betas = {}
        
        for benchmark_col in benchmark_cols:
            if benchmark_col in port_ret.columns:
                benchmark = port_ret[benchmark_col].reindex(returns.index).dropna()
                aligned_returns = returns.loc[benchmark.index]
                
                if len(aligned_returns) > 0 and not benchmark.isnull().all():
                    X = sm.add_constant(benchmark)
                    reg = sm.OLS(aligned_returns, X).fit()
                    try:
                        alphas[benchmark_col] = reg.params['const']
                        betas[benchmark_col] = reg.params[benchmark_col]
                    except (KeyError, IndexError):
                        alphas[benchmark_col] = reg.params.iloc[0]
                        betas[benchmark_col] = reg.params.iloc[1]
                else:
                    alphas[benchmark_col] = np.nan
                    betas[benchmark_col] = np.nan
            else:
                alphas[benchmark_col] = np.nan
                betas[benchmark_col] = np.nan

        # VaR and Hit Ratio
        var5 = np.percentile(returns, 5)
        hit = (returns > 0).mean()

        # Fill performance table
        perf.at['Geometric Mean Return', col] = geo_mean
        perf.at['Downside Deviation', col] = dd
        perf.at['Max Drawdown', col] = drawdown
        perf.at['Sortino Ratio', col] = sortino
        perf.at['Skewness', col] = skewness
        perf.at['Kurtosis', col] = kurt
        perf.at['Alpha (vs SPX)', col] = alphas.get('SPX', np.nan)
        perf.at['Beta (vs SPX)', col] = betas.get('SPX', np.nan)
        perf.at['Alpha (vs SAMPLE_VW)', col] = alphas.get('SAMPLE_VW', np.nan)
        perf.at['Beta (vs SAMPLE_VW)', col] = betas.get('SAMPLE_VW', np.nan)
        perf.at['VaR 5%', col] = var5
        perf.at['Hit Ratio', col] = hit

    return perf

# Update performance table with both benchmarks
perf_table_vw = portfolio_performance_table(port_rets_vw, benchmark_cols=['SPX', 'SAMPLE_VW'])
perf_table_vw = perf_table_vw.map(lambda x: f"{x:.4f}" if isinstance(x, (float, np.floating)) else x)
print("\n=== Performance Table (with SPX and SAMPLE_VW Benchmarks) ===")
print(perf_table_vw)

# For equal-weighted:
perf_table_ew = portfolio_performance_table(port_rets_eq, benchmark_col='SPW')
perf_table_ew = perf_table_ew.map(lambda x: f"{x:.4f}" if isinstance(x, (float, np.floating)) else x)
print(perf_table_ew)



file4_path = "C:\\Users\\thkraft\\eCommerce-Goethe Dropbox\\Thilo Kraft\\Thilo(privat)\\Privat\\Research\\RRR_FinancialImplication\\Data\\2025-06-27-FF-Factors_Monthly_Python.csv"

# Read Fama-French monthly factors, skipping metadata
ff_factors = pd.read_csv(file4_path, skiprows=3)
ff_factors = ff_factors.rename(columns={'Unnamed: 0': 'Date'})
ff_factors['Date'] = pd.to_datetime(ff_factors['Date'].astype(str), format='%Y%m')
ff_factors = ff_factors.set_index('Date').sort_index()
for col in ['Mkt-RF', 'SMB', 'HML', 'RF']:
    ff_factors[col] = ff_factors[col] / 100

# Align index to month end if needed
ff_factors.index = ff_factors.index.to_period('M').to_timestamp('M')

# Make sure both DataFrames have the same index frequency/type
port_rets_vw.index = port_rets_vw.index.to_period('M').to_timestamp('M')
combined = port_rets_vw.join(ff_factors, how='inner')
combined.head()

import statsmodels.api as sm



# Create long-short portfolio (Q4 - Q1)
combined['Q4-Q1'] = combined['Q4'] - combined['Q1']

# Choose which portfolios to analyze (include long-short)
portfolios = [col for col in port_rets_vw.columns if col not in ['SPX', 'SPW', 'SAMPLE_VW']]
portfolios.append('Q4-Q1')  # Add the long-short portfolio

print("\n" + "="*80)
print("FAMA-FRENCH 3-FACTOR REGRESSIONS")
print("="*80)

# Store results for comparison
ff_results = {}

for p in portfolios:
    # Excess returns (portfolio minus risk-free)
    y = combined[p] - combined['RF']
    
    # Explanatory variables (factors)
    X = combined[['Mkt-RF', 'SMB', 'HML']]
    X = sm.add_constant(X)
    
    # Run regression
    model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 3})  # Newey-West standard errors
    
    print(f'\n{"="*60}')
    print(f'Fama-French Regression for {p}:')
    print(f'{"="*60}')
    print(model.summary())
    
    # Store key metrics
    ff_results[p] = {
        'Alpha': model.params['const'],
        'Alpha_tstat': model.tvalues['const'],
        'Alpha_pval': model.pvalues['const'],
        'Beta_Mkt': model.params['Mkt-RF'],
        'Beta_SMB': model.params['SMB'],
        'Beta_HML': model.params['HML'],
        'R-squared': model.rsquared,
        'Adj_R-squared': model.rsquared_adj
    }

# Create summary table of all results
ff_summary = pd.DataFrame(ff_results).T
print("\n" + "="*80)
print("FAMA-FRENCH REGRESSION SUMMARY TABLE")
print("="*80)
print(ff_summary.round(4))



'''
neu ende
'''

'''
update with revenue acquisitn
# =============================================================================
# 0) Imports
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# 1) Read monthly PX_LAST file (two header rows) and build monthly log returns
#    NOTE: if you've already built df_monthly_returns, you can skip to step 2.
# =============================================================================
file3_path = r"C:\Users\thkraft\eCommerce-Goethe Dropbox\Thilo Kraft\Thilo(privat)\Privat\Research\RRR_FinancialImplication\Data\2025-07-02a-TK-Monthly-Returns_Python.xlsx"

# read with no header, then compose headers from first 2 rows
monthly_raw = pd.read_excel(file3_path, header=None)
header_rows = monthly_raw.iloc[:2]
data_rows   = monthly_raw.iloc[2:].copy()

# build "Firm.Variable" headers (e.g., "SPX INDEX.PX_LAST")
combined_headers = header_rows.apply(lambda x: x.astype(str).str.strip(), axis=0)
col_headers = combined_headers.apply(lambda x: '.'.join(x.dropna()), axis=0)
data_rows.columns = col_headers

# rename first column to Date and parse
data_rows = data_rows.reset_index(drop=True)
data_rows.rename(columns={data_rows.columns[0]: 'Date'}, inplace=True)
data_rows.replace('#N/A N/A', np.nan, inplace=True)

# parse Date (your file looks like DD/MM/YYYY)
data_rows['Date'] = pd.to_datetime(data_rows['Date'], errors='coerce', dayfirst=True)

# ensure numeric for all PX columns
df_prices = data_rows.copy()
for c in df_prices.columns:
    if c != 'Date':
        df_prices[c] = pd.to_numeric(df_prices[c], errors='coerce')

# monthly log returns
px_only = df_prices.drop(columns=['Date'])
returns_px = np.log(px_only / px_only.shift(1))

# back the Date column in & melt to long
returns_px['Date'] = df_prices['Date'].values
returns = returns_px.melt(id_vars='Date', var_name='FirmVar', value_name='RETURN_LOG').dropna(subset=['RETURN_LOG'])

# extract firm name (strip the ".PX_LAST" suffix)
returns['FIRM'] = returns['FirmVar'].str.replace('.PX_LAST', '', regex=False)
returns.drop(columns=['FirmVar'], inplace=True)

# unify firm casing for safe merges
returns['FIRM'] = returns['FIRM'].str.upper()

# build quarter end key for monthly dates
returns['QUARTER'] = returns['Date'].dt.to_period('Q').dt.to_timestamp('Q', 'end')

# =============================================================================
# 2) Prepare quarterly signals from df_long (RRR_LAG, MCAP, #Acq_Rate)
#    Assumes df_long exists with MultiIndex ['FIRM','DATE'] and these columns.
# =============================================================================
# if your df_long does not have 'RRR_LAG' but has 'RRR_pct_lag1' (in %), create it:
if 'RRR_LAG' not in df_long.columns and 'RRR_pct_lag1' in df_long.columns:
    df_long['RRR_LAG'] = pd.to_numeric(df_long['RRR_pct_lag1'], errors='coerce') / 100.0

df_q = (
    df_long.reset_index()[['FIRM', 'DATE', 'RRR_LAG', 'HISTORICAL_MARKET_CAP', '#Acq_Rate']]
    .copy()
)
df_q['FIRM'] = df_q['FIRM'].astype(str).str.upper()
df_q['DATE'] = pd.to_datetime(df_q['DATE'], errors='coerce')
df_q.rename(columns={'#Acq_Rate': 'REV_NEW_G'}, inplace=True)
df_q['QUARTER'] = df_q['DATE'].dt.to_period('Q').dt.to_timestamp('Q', 'end')

# =============================================================================
# 3) Merge quarterly signals onto monthly returns (by FIRM, QUARTER)
# =============================================================================
returns = returns.merge(
    df_q[['FIRM', 'QUARTER', 'RRR_LAG', 'HISTORICAL_MARKET_CAP', 'REV_NEW_G']],
    on=['FIRM','QUARTER'],
    how='left'
)

# =============================================================================
# 4) Assign quartiles each quarter for BOTH sorts:
#       (a) RRR_LAG-based quartiles  → column: 'quartile'
#       (b) New-Rev-Growth quartiles → column: 'quartile_new'
# =============================================================================
def assign_qcut(series, q=4, labels=None):
    """Return a Series aligned to 'series' index with quartile labels; NaN if not enough spread."""
    labels = labels or [f"Q{i}" for i in range(1, q+1)]
    s = pd.to_numeric(series, errors='coerce')
    out = pd.Series(index=s.index, dtype=object)
    mask = s.notna()
    s_non = s[mask]
    if s_non.nunique() < q:
        return out  # all NaN
    try:
        out.loc[mask] = pd.qcut(s_non, q=q, labels=labels, duplicates='drop').astype(object).values
    except ValueError:
        # ties / insufficient spread after duplicates='drop'
        return pd.Series(index=s.index, dtype=object)
    return out

returns['RRR_LAG']  = pd.to_numeric(returns['RRR_LAG'], errors='coerce')
returns['REV_NEW_G'] = pd.to_numeric(returns['REV_NEW_G'], errors='coerce')

# RRR-based quartiles
returns['quartile'] = (
    returns.groupby('QUARTER')['RRR_LAG']
           .transform(lambda s: assign_qcut(s, q=4, labels=['Q1','Q2','Q3','Q4']))
)

# New-Rev-Growth-based quartiles
returns['quartile_new'] = (
    returns.groupby('QUARTER')['REV_NEW_G']
           .transform(lambda s: assign_qcut(s, q=4, labels=['Q1','Q2','Q3','Q4']))
)

# =============================================================================
# 5) Function to compute monthly portfolio returns (equal/value weighted),
#    add benchmarks from the same 'returns' table, cumulate & plot.
# =============================================================================
def calc_and_plot_portfolio_returns_from_long(
    returns: pd.DataFrame,
    weight_type: str = 'value',                     # 'value' or 'equal'
    benchmarks = ('SPX INDEX', 'SPW INDEX'),        # names in returns['FIRM']
    quartile_col: str = 'quartile',                 # 'quartile' or 'quartile_new'
    title_prefix: str = 'RRR Quartile'
):
    """
    Input 'returns' must have columns:
      Date, FIRM, RETURN_LOG, QUARTER, quartile columns, HISTORICAL_MARKET_CAP
    Produces monthly portfolio log-return series for Q1..Q4 + specified benchmarks.
    """
    df = returns.copy()

    # portfolio formation
    if weight_type.lower() == 'equal':
        port_rets = (
            df.groupby(['Date', quartile_col])['RETURN_LOG']
              .mean()
              .unstack(quartile_col)
              .sort_index()
        )
    else:
        df['MCAP'] = pd.to_numeric(df['HISTORICAL_MARKET_CAP'], errors='coerce')
        df['MCAP_SUM'] = df.groupby(['Date', quartile_col])['MCAP'].transform('sum')
        df['w'] = df['MCAP'] / df['MCAP_SUM']
        df['w_return'] = df['w'] * df['RETURN_LOG']
        port_rets = (
            df.groupby(['Date', quartile_col])['w_return']
              .sum()
              .unstack(quartile_col)
              .sort_index()
        )

    # benchmarks: pull directly from returns (monthly log returns already)
    for bmk in benchmarks:
        bmk_up = str(bmk).upper()
        if bmk_up in df['FIRM'].unique():
            ser = (df[df['FIRM'] == bmk_up]
                   .groupby('Date')['RETURN_LOG']
                   .mean())  # in case of duplicates
            port_rets[bmk_up] = ser.reindex(port_rets.index)

    # cumulative (start at 0: i.e., growth from 0 → cum = exp(cumsum)-1)
    cum = np.exp(port_rets.cumsum()) - 1.0
    # prepend a zero month for a clean start at 0
    if not cum.empty:
        first_date = cum.index.min()
        prior_date = (first_date - pd.offsets.MonthEnd(1))
        start_row = pd.DataFrame({c: 0.0 for c in cum.columns}, index=[prior_date])
        cum = pd.concat([start_row, cum]).sort_index()

    # plot
    plt.figure(figsize=(12,6))
    for col in cum.columns:
        plt.plot(cum.index, cum[col], label=col)
    plt.title(f'{title_prefix} Portfolios vs Benchmarks ({weight_type.capitalize()}-Weighted)')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return (start = 0)')
    plt.legend(title='Series', loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    return port_rets, cum

# =============================================================================
# 6) RUN: RRR-sorted (quartile), value-weighted & equal-weighted
# =============================================================================
port_rets_vw_RRR, cum_vw_RRR = calc_and_plot_portfolio_returns_from_long(
    returns, weight_type='value',
    benchmarks=('SPX INDEX', 'SPW INDEX'),
    quartile_col='quartile',
    title_prefix='RRR'
)

port_rets_eq_RRR, cum_eq_RRR = calc_and_plot_portfolio_returns_from_long(
    returns, weight_type='equal',
    benchmarks=('SPX INDEX', 'SPW INDEX'),
    quartile_col='quartile',
    title_prefix='RRR'
)

# =============================================================================
# 7) RUN: New-Revenue-Growth-sorted (quartile_new), value- & equal-weighted
# =============================================================================
port_rets_vw_NEW, cum_vw_NEW = calc_and_plot_portfolio_returns_from_long(
    returns, weight_type='value',
    benchmarks=('SPX INDEX', 'SPW INDEX'),
    quartile_col='quartile_new',
    title_prefix='New-Rev-Growth'
)

port_rets_eq_NEW, cum_eq_NEW = calc_and_plot_portfolio_returns_from_long(
    returns, weight_type='equal',
    benchmarks=('SPX INDEX', 'SPW INDEX'),
    quartile_col='quartile_new',
    title_prefix='New-Rev-Growth'
)


'''
#==============================================================================
# 6. Portfolio Analysis
#==============================================================================


###creating portfolios
###creating portfolios
###creating portfolios
def quartile_cumulative_returns(
    df_long: pd.DataFrame,
    weight_by_mcap: bool = True
):
    """
    1) Assign each firm to one of four RRR quartiles per quarter (using RRR_LAG).
    2) Compute mean log return (RETURN_LOG) for each quartile per quarter,
       optionally weighting by lagged market cap.
    3) Add the SPX INDEX log return.
    4) Build a DataFrame of cumulative returns (start = 1 at a synthetic prior date).
    5) Plot Q1–Q4 and SPX cumulative return curves on one graph, with a “$1” start row.
    """

    # 1) Flatten index to assign quartiles
    df = df_long.reset_index()

    # 1a) Drop rows missing RRR_LAG or RETURN_LOG (and MCAP if weighting)
    required = ['RRR_LAG', 'RETURN_LOG']
    if weight_by_mcap:
        required.append('HISTORICAL_MARKET_CAP')
    df = df.dropna(subset=required)

    # 1b) Convert RRR_LAG to numeric
    df['RRR_LAG'] = pd.to_numeric(df['RRR_LAG'], errors='coerce')

    # 1c) Assign quartile (Q1–Q4) per DATE
    df['quartile'] = df.groupby('DATE')['RRR_LAG'].transform(
        lambda x: pd.qcut(x.astype(float), q=4,
                          labels=['Q1','Q2','Q3','Q4'],
                          duplicates='drop')
    )

    # 2) Compute mean (or weighted) log-return per DATE × quartile
    if not weight_by_mcap:
        # simple mean
        mean_quartile = (
            df
            .groupby(['DATE','quartile'])['RETURN_LOG']
            .mean()
            .unstack('quartile')
            .sort_index()
        )
    else:
        # value-weighted by HISTORICAL_MARKET_CAP at same DATE
        df['MCAP'] = pd.to_numeric(df['HISTORICAL_MARKET_CAP'], errors='coerce')
        # sum of MCAP per (DATE, quartile)
        df['MCAP_SUM'] = df.groupby(['DATE','quartile'])['MCAP'].transform('sum')
        # weight_i = MCAP_i / MCAP_SUM
        df['w'] = df['MCAP'] / df['MCAP_SUM']
        # weighted average of RETURN_LOG
        mean_quartile = (
            df
            .assign(weighted_ret=lambda x: x['w'] * x['RETURN_LOG'])
            .groupby(['DATE','quartile'])['weighted_ret']
            .sum()
            .unstack('quartile')
            .sort_index()
        )

    # Ensure columns Q1..Q4 exist (in case any quartile dropped)
    for q in ['Q1','Q2','Q3','Q4']:
        if q not in mean_quartile.columns:
            mean_quartile[q] = np.nan
    mean_quartile = mean_quartile[['Q1','Q2','Q3','Q4']]

    # 3) SPX log-return, aligned to same index
    spx = df_long.xs('SPX INDEX', level='FIRM')['RETURN_LOG']
    spx = spx.reindex(mean_quartile.index)

    # 4) Combine into mean_returns (log-return)
    mean_returns = mean_quartile.copy()
    mean_returns['SPX'] = spx

    # 5) Cumulative log‐sum then exponentiate → cumulative return factor
    log_cum = mean_returns.cumsum()
    cum_returns = np.exp(log_cum) -1

    # 6) Prepend a “start” row of 1.0 at one quarter before the first real date
    first_date = cum_returns.index.min()
    prior_date = first_date - pd.DateOffset(months=3)
    start_row = pd.DataFrame(
        {col: 0.0 for col in cum_returns.columns},
        index=[prior_date]
    )
    cum_with_start = pd.concat([start_row, cum_returns]).sort_index()

    # 7) Plot all series, including the fake start
    plt.figure(figsize=(10, 6))
    for col in ['Q1','Q2','Q3','Q4','SPX']:
        plt.plot(cum_with_start.index, cum_with_start[col], label=col)
    plt.title('Cumulative Returns: Q1–Q4 Portfolios vs SPX Index')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend(title='Series', loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    return cum_with_start



# Value‐weighted by market cap:
cum_vw = quartile_cumulative_returns(df_long, weight_by_mcap=True)
cum_vw.head()
# equally weighted:
'''
change to equally weighted S&P index
'''
cum_eq = quartile_cumulative_returns(df_long, weight_by_mcap=False)
cum_eq.head()

###analyzing portfolios with performance metrics
###analyzing portfolios with performance metrics
###analyzing portfolios with performance metrics

import statsmodels.api as sm
from scipy.stats import skew, kurtosis

def portfolio_performance_table(port_ret, benchmark_col='SPX'):
    measures = [
        'Geometric Mean Return', 'Downside Deviation', 'Max Drawdown',
        'Sortino Ratio', 'Skewness', 'Kurtosis', 'Alpha', 'Beta',
        'VaR 5%', 'Hit Ratio'
    ]
    portfolios = list(port_ret.columns)
    perf = pd.DataFrame(index=measures, columns=portfolios)
    
    for col in portfolios:
        returns = port_ret[col].dropna()
        benchmark = port_ret[benchmark_col].reindex(returns.index).dropna()
        returns = returns.loc[benchmark.index]  # align

        # Geometric mean
        geo_mean = np.exp(np.log1p(returns).mean()) - 1 #np.log1p(x) is equivalent to np.log(1 + x) offering improved numerical stability for small values

        # Downside deviation (negative returns only)
        downside = returns[returns < 0]
        dd = downside.std(ddof=0)  # population std for downside

        # Max drawdown
        cum = (1 + returns).cumprod()
        cum_max = cum.cummax() #Return a df or series of the same size containing cumulative maximum 
        drawdown = (cum / cum_max - 1).min()

        # Sortino ratio
        sortino = returns.mean() / dd if dd > 0 else np.nan

        # Skewness and kurtosis
        skewness = skew(returns, nan_policy='omit')
        kurt = kurtosis(returns, nan_policy='omit', fisher=False)  # "normal" = 3

        # Alpha and Beta (CAPM regression vs. SPX)
        X = sm.add_constant(benchmark)
        reg = sm.OLS(returns, X).fit()
        alpha = reg.params['const']
        beta = reg.params[benchmark_col]

        # 5% Value at Risk (VaR, historical)
        var5 = np.percentile(returns, 5)

        # Hit ratio (fraction positive returns)
        hit = (returns > 0).mean()

        perf.at['Geometric Mean Return', col] = geo_mean
        perf.at['Downside Deviation', col] = dd
        perf.at['Max Drawdown', col] = drawdown
        perf.at['Sortino Ratio', col] = sortino
        perf.at['Skewness', col] = skewness
        perf.at['Kurtosis', col] = kurt
        perf.at['Alpha', col] = alpha
        perf.at['Beta', col] = beta
        perf.at['VaR 5%', col] = var5
        perf.at['Hit Ratio', col] = hit

    return perf


port_ret = cum_vw.diff().dropna()  # Each column: Q1, Q2, Q3, Q4, SPX
perf_table = portfolio_performance_table(port_ret)
perf_table = perf_table.map(
    lambda x: f"{x:.4f}" if isinstance(x, (float, np.floating)) else x
)
print(perf_table)

'''
change to equally weighted SPX index
'''
port_ret_eq = cum_eq.diff().dropna()  # Each column: Q1, Q2, Q3, Q4, SPX
perf_table_eq = portfolio_performance_table(port_ret_eq)
perf_table_eq = perf_table_eq.map(  
    lambda x: f"{x:.4f}" if isinstance(x, (float, np.floating)) else x
)   
print(perf_table_eq)





###analyzing portfolios with F&F
###analyzing portfolios with F&F
###analyzing portfolios with F&F


# factors are available monthly or annually
# use monthly factors 
# align portfolio time series returns to the factor dates
# merge your returns with the factor series on the monthly data 
# run the regression: portfolioreturn -RF = alpha + beta1 * MKT_RF + beta2 * SMB + beta3 * HML

# factors are available at https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
# Download the Fama-French factors data (e.g., 5 factors) and save it as a CSV file

# Header is on line 5 (row 4), data starts on line 6 (row 5)
file4_path = "C:\\Users\\thkraft\\eCommerce-Goethe Dropbox\\Thilo Kraft\\Thilo(privat)\\Privat\\Research\\RRR_FinancialImplication\\Data\\2025-06-27-FF_Factors.csv"

#manually delete annual factors after the monthly factors
try:
    # Read the fundamentals file
    ff_factors = pd.read_csv(file4_path, skiprows=3)
    monthly_returns = pd.read_excel(file4_path, sheet_name='Monthly Returns')
    #ff_factors = pd.read_csv(file3_path, skiprows=4,index_col=0)
    print("Fama and French Fundamentals-Daten erfolgreich eingelesen.")
    print("Shape:", ff_factors.shape)

except Exception as e:
    #Exception is the base class for all exceptions (FilenotFoundError, ValueError, etc.)
    #As e saves the "exception"
    print(f"Fehler beim Einlesen der Dateien: {e}")
ff_factors.head()
# Rename 'Unnamed: 0' to 'Date' for clarity
ff_factors = ff_factors.rename(columns={'Unnamed: 0': 'Date'})

# Convert date to datetime (YYYYMM format)
ff_factors['Date'] = pd.to_datetime(ff_factors['Date'].astype(str), format='%Y%m')
# Set as index and sort
ff_factors = ff_factors.set_index('Date').sort_index()
for col in ['Mkt-RF', 'SMB', 'HML', 'RF']:
    ff_factors[col] = ff_factors[col] / 100
ff_factors.head()



#way two: interpolate quarterly returns to monthly returns
quarter_rets = cum_vw.diff().dropna()
# Let's assume the index of quarter_rets is quarterly, e.g., 2020-03-31, 2020-06-30, etc.
monthly_rets = []
monthly_dates = []

for date, row in quarter_rets.iterrows():
    r_q = row.values
    r_m = (1 + r_q) ** (1/3) - 1  # Decompound
    # Add three months for each quarter
    for i in range(3):
        month_date = (pd.to_datetime(date) - pd.offsets.MonthEnd(0)) + pd.DateOffset(months=i-2)
        monthly_rets.append(r_m)
        monthly_dates.append(month_date)

monthly_df = pd.DataFrame(monthly_rets, columns=quarter_rets.columns, index=pd.to_datetime(monthly_dates))
monthly_df = monthly_df.sort_index()
monthly_df.head()

# Set both to month-end frequency for perfect alignment
monthly_df.index = monthly_df.index.to_period('M').to_timestamp('M')
ff_factors.index = ff_factors.index.to_period('M').to_timestamp('M')
combined = monthly_df.join(ff_factors, how='inner')
combined.head()

portfolios = [col for col in monthly_df.columns if col != 'SPX']

for p in portfolios:
    # Excess returns: subtract risk-free rate (ensure both in percent or decimal!)
    y = combined[p] - combined['RF'] / 100   # if RF is in percent and your returns are decimal
    X = combined[['Mkt-RF', 'SMB', 'HML']] / 100  # if these are in percent
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    print(f'\nFama-French regression for {p}:')
    print(model.summary())

#==============================================================================
# %%
