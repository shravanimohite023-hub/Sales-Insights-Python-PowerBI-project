import pandas as pd
import numpy  as np
data=pd.read_csv(r"C:\Users\STORMSOFTS\Desktop\messy.csv")
##########shape (to find row,columns number)#########################
print(data.shape)
###############remove duplicate#############################
print(data.duplicated().sum())
############### drop duplicate (remove)################
df=data.drop_duplicates()
print(df.shape)
#######summery(EDA)######################
print(df.describe())
############## missing value ##################
numeric_cols=data.select_dtypes(include="number").columns
for col in numeric_cols:
    Q1= data[col].quantile(0.25)
    Q3= data[col].quantile(0.75)

    IQR=Q3-Q1
    Lower=Q1-1.5*IQR
    Upper=Q3+1.5*IQR
    Outlier = data [(data[col]<Lower)|(data[col]>Upper)]
    print("\ncolumns:",col)
    print("Lower limit:",Lower)
    print("Upper limit:",Upper)
    print("number of outliers")
len((Outlier))
data["quantity"]=data["quantity"].fillna(data["quantity"].median())
data["unit_cost"]=data["unit_cost"].fillna(data["unit_cost"].mean())
data["unit_price"]=data["unit_price"].fillna(data["unit_price"].mean())
data["revenue"]=data["revenue"].fillna(data["revenue"].median())
data[ "cost"]=data["cost"].fillna(data["cost"].median())
data["margin"]=data["margin"].fillna(data["margin"].median())
print("\nafter filling")    
print(data.isnull().sum())
#############remove the row####################
df=df.dropna(subset=["order_date","revenue","cost"])
#################profit####################
df["profit"]=df["revenue"]-df["cost"]

df["profit_margin_pct"]=np.where(df["revenue"]!=0,(df["profit"]/df["revenue"])*100,0)  
print(df.shape)
#################shiping days####################
df["order_date"]=pd.to_datetime(df["order_date"],errors="coerce")
#############################year####################################
df["year"]=df["order_date"].dt.year
df["month_name"]=df["order_date"].dt.strftime("%B")
df["year_month"]=df["order_date"].dt.to_period("M").astype(str)
df["quarter"]=df["order_date"].dt.to_period("Q").astype(str)
df["day"]=df["order_date"].dt.day
df["weekday"]=df["order_date"].dt.day_name()
print(df.columns)
# 6. OVERALL SALES KPIs
# ============================================================

total_revenue = df["revenue"].sum()
total_cost = df["cost"].sum()
total_profit = df["profit"].sum()
total_quantity = df["quantity"].sum()

total_orders = df["invoice_no"].nunique()
total_customers = df["customer_id"].nunique()

average_order_value = (
    total_revenue / total_orders
    if total_orders != 0 else 0
)

overall_margin = (
    (total_profit / total_revenue) * 100
    if total_revenue != 0 else 0
)

print("\n======================================")
print("           SALES KPIs")
print("======================================")

print(f"Total Revenue      : {total_revenue:,.2f}")
print(f"Total Cost         : {total_cost:,.2f}")
print(f"Total Profit       : {total_profit:,.2f}")
print(f"Total Quantity     : {total_quantity:,.0f}")
print(f"Total Orders       : {total_orders:,}")
print(f"Total Customers    : {total_customers:,}")
print(f"Average Order Value: {average_order_value:,.2f}")
print(f"Profit Margin      : {overall_margin:.2f}%")


# ============================================================
# 7. MONTHLY SALES ANALYSIS
# ============================================================

monthly_sales = (
    df.groupby("year_month")
    .agg(
        revenue=("revenue", "sum"),
        cost=("cost", "sum"),
        profit=("profit", "sum"),
        quantity=("quantity", "sum"),
        orders=("invoice_no", "nunique")
    )
    .reset_index()
)

# Month-over-Month Growth
monthly_sales["revenue_growth_pct"] = (
    monthly_sales["revenue"].pct_change() * 100
)

print("\n========== MONTHLY SALES ==========")
print(monthly_sales)


# ============================================================
# 8. YEARLY ANALYSIS
# ============================================================

yearly_sales = (
    df.groupby("year")
    .agg(
        revenue=("revenue", "sum"),
        cost=("cost", "sum"),
        profit=("profit", "sum"),
        quantity=("quantity", "sum"),
        orders=("invoice_no", "nunique")
    )
    .reset_index()
)

print("\n========== YEARLY SALES ==========")
print(yearly_sales)


# ============================================================
# 9. QUARTERLY ANALYSIS
# ============================================================

quarterly_sales = (
    df.groupby("quarter")
    .agg(
        revenue=("revenue", "sum"),
        profit=("profit", "sum"),
        quantity=("quantity", "sum")
    )
    .reset_index()
)

print("\n========== QUARTERLY SALES ==========")
print(quarterly_sales)


# ============================================================
# 10. TOP CUSTOMERS BY REVENUE
# ============================================================

top_customers_revenue = (
    df.groupby(["customer_id", "customer"])
    .agg(
        revenue=("revenue", "sum"),
        profit=("profit", "sum"),
        quantity=("quantity", "sum"),
        orders=("invoice_no", "nunique")
    )
    .sort_values("revenue", ascending=False)
    .reset_index()
)

print("\n========== TOP 10 CUSTOMERS BY REVENUE ==========")
print(top_customers_revenue.head(10))


# ============================================================
# 11. TOP CUSTOMERS BY PROFIT
# ============================================================

top_customers_profit = (
    df.groupby(["customer_id", "customer"])
    .agg(
        revenue=("revenue", "sum"),
        profit=("profit", "sum"),
        orders=("invoice_no", "nunique")
    )
    .sort_values("profit", ascending=False)
    .reset_index()
)

print("\n========== TOP 10 CUSTOMERS BY PROFIT ==========")
print(top_customers_profit.head(10))

# 12. PRODUCT ANALYSIS
# ============================================================

product_analysis = (
    df.groupby(["product_id", "product"])
    .agg(
        revenue=("revenue", "sum"),
        cost=("cost", "sum"),
        profit=("profit", "sum"),
        quantity=("quantity", "sum"),
        orders=("invoice_no", "nunique")
    )
    .reset_index()
)

product_analysis["margin_pct"] = (
    product_analysis["profit"] /
    product_analysis["revenue"] * 100
).replace([np.inf, -np.inf], 0).fillna(0)


print("\n========== TOP 10 PRODUCTS BY REVENUE ==========")
print(
    product_analysis
    .sort_values("revenue", ascending=False)
    .head(10)
)

print("\n========== TOP 10 PRODUCTS BY PROFIT ==========")
print(
    product_analysis
    .sort_values("profit", ascending=False)
    .head(10)
)

print("\n========== TOP 10 PRODUCTS BY QUANTITY ==========")
print(
    product_analysis
    .sort_values("quantity", ascending=False)
    .head(10)
)


# ============================================================
# 13. LOW PERFORMING PRODUCTS
# ============================================================

print("\n========== LOWEST REVENUE PRODUCTS ==========")

print(
    product_analysis
    .sort_values("revenue", ascending=True)
    .head(10)
)



# 14. HIGHEST MARGIN PRODUCTS #######################################


print("\n========== HIGHEST MARGIN PRODUCTS ==========")

print(
    product_analysis
    .sort_values("margin_pct", ascending=False)
    .head(10)
)


# ============================================================
# 15. CATEGORY ANALYSIS
# ============================================================

category_analysis = (
    df.groupby("category")
    .agg(
        revenue=("revenue", "sum"),
        cost=("cost", "sum"),
        profit=("profit", "sum"),
        quantity=("quantity", "sum"),
        orders=("invoice_no", "nunique")
    )
    .reset_index()
)

category_analysis["margin_pct"] = (
    category_analysis["profit"] /
    category_analysis["revenue"] * 100
)

print("\n========== CATEGORY ANALYSIS ==========")
print(
    category_analysis
    .sort_values("revenue", ascending=False)
)


# ============================================================
# 16. SEGMENT ANALYSIS
# ============================================================

segment_analysis = (
    df.groupby("segment")
    .agg(
        revenue=("revenue", "sum"),
        cost=("cost", "sum"),
        profit=("profit", "sum"),
        quantity=("quantity", "sum"),
        orders=("invoice_no", "nunique"),
        customers=("customer_id", "nunique")
    )
    .reset_index()
)

segment_analysis["margin_pct"] = (
    segment_analysis["profit"] /
    segment_analysis["revenue"] * 100
)

print("\n========== SEGMENT ANALYSIS ==========")
print(segment_analysis)

# 17. CATEGORY + SEGMENT ANALYSIS
# ============================================================

category_segment = (
    df.groupby(["category", "segment"])
    .agg(
        revenue=("revenue", "sum"),
        profit=("profit", "sum"),
        quantity=("quantity", "sum")
    )
    .reset_index()
)

print("\n========== CATEGORY + SEGMENT ==========")
print(category_segment)


# ============================================================
# 20. CUSTOMER CONTRIBUTION
# ============================================================

customer_revenue = (
    df.groupby("customer_id")["revenue"]
    .sum()
    .sort_values(ascending=False)
)

top_20_percent_count = max(
    1,
    int(len(customer_revenue) * 0.20)
)

top_20_revenue = customer_revenue.head(
    top_20_percent_count
).sum()

customer_contribution = (
    top_20_revenue / total_revenue * 100
)

print("\n========== PARETO ANALYSIS ==========")

print(
    f"Top 20% customers contribute "
    f"{customer_contribution:.2f}% of total revenue."
)
# 22. HIGH PROFIT PRODUCTS
# ============================================================

high_profit_products = product_analysis[
    product_analysis["profit"] >=
    product_analysis["profit"].quantile(0.75)
]

print("\n========== HIGH PROFIT PRODUCTS ==========")

print(
    high_profit_products
    .sort_values("profit", ascending=False)
)
# 22. HIGH PROFIT PRODUCTS
# ============================================================

high_profit_products = product_analysis[
    product_analysis["profit"] >=
    product_analysis["profit"].quantile(0.75)
]

print("\n========== HIGH PROFIT PRODUCTS ==========")

print(
    high_profit_products
    .sort_values("profit", ascending=False)
)
# 21. HIGH REVENUE BUT LOW MARGIN PRODUCTS
# ============================================================

high_revenue_products = product_analysis[
    product_analysis["revenue"] >=
    product_analysis["revenue"].quantile(0.75)
]

low_margin_high_revenue = high_revenue_products[
    high_revenue_products["margin_pct"] <=
    product_analysis["margin_pct"].median()
]

print("\n========== HIGH REVENUE + LOW MARGIN PRODUCTS ==========")

print(
    low_margin_high_revenue
    .sort_values("revenue", ascending=False)
)
# 23. DAILY SALES TREND
# ============================================================

daily_sales = (
    df.groupby("order_date")
    .agg(
        revenue=("revenue", "sum"),
        profit=("profit", "sum")
    )
    .reset_index()
)

print("\n========== DAILY SALES ==========")
print(daily_sales.head())
# 24. WEEKDAY ANALYSIS
# ============================================================

weekday_analysis = (
    df.groupby("weekday")
    .agg(
        revenue=("revenue", "sum"),
        profit=("profit", "sum"),
        quantity=("quantity", "sum")
    )
    .reset_index()
)

print("\n========== WEEKDAY ANALYSIS ==========")
print(weekday_analysis)
# 25. VISUALIZATION
# ============================================================


# -----------------------------
# Monthly Revenue
# -----------------------------
import matplotlib.pyplot as plt

plt.plot(
    monthly_sales["year_month"],
    monthly_sales["revenue"],
    marker="o"
)

plt.title("Monthly Revenue Trend")
plt.xlabel("Month")
plt.ylabel("Revenue")
plt.xticks(rotation=45)
plt.grid(True)
plt.show()
# Monthly Profit
# -----------------------------

plt.figure(figsize=(12, 6))

plt.plot(
    monthly_sales["year_month"],
    monthly_sales["profit"],
    marker="o"
)

plt.title("Monthly Profit Trend")
plt.xlabel("Month")
plt.ylabel("Profit")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()


# -----------------------------
# Top 10 Products
# -----------------------------

top_products = (
    product_analysis
    .sort_values("revenue", ascending=False)
    .head(10)
)

plt.figure(figsize=(12, 6))

plt.barh(
    top_products["product"],
    top_products["revenue"]
)

plt.title("Top 10 Products by Revenue")
plt.xlabel("Revenue")
plt.ylabel("Product")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()


# -----------------------------
# Category Revenue
# -----------------------------

plt.figure(figsize=(10, 6))

plt.bar(
    category_analysis["category"],
    category_analysis["revenue"]
)

plt.title("Revenue by Category")
plt.xlabel("Category")
plt.ylabel("Revenue")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# -----------------------------
# Category Profit
# -----------------------------

plt.figure(figsize=(10, 6))

plt.bar(
    category_analysis["category"],
    category_analysis["profit"]
)

plt.title("Profit by Category")
plt.xlabel("Category")
plt.ylabel("Profit")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# -----------------------------
# Segment Revenue
# -----------------------------

plt.figure(figsize=(8, 5))

plt.bar(
    segment_analysis["segment"],
    segment_analysis["revenue"]
)

plt.title("Revenue by Segment")
plt.xlabel("Segment")
plt.ylabel("Revenue")
plt.tight_layout()
plt.show()


# -----------------------------
# Revenue vs Profit
# -----------------------------

plt.figure(figsize=(10, 6))

plt.scatter(
    product_analysis["revenue"],
    product_analysis["profit"]
)

plt.title("Product Revenue vs Profit")
plt.xlabel("Revenue")
plt.ylabel("Profit")
plt.grid(True)
plt.tight_layout()
plt.show()
# 26. FINAL BUSINESS INSIGHTS
# ============================================================

best_product = product_analysis.loc[
    product_analysis["revenue"].idxmax()
]

most_profitable_product = product_analysis.loc[
    product_analysis["profit"].idxmax()
]

best_category = category_analysis.loc[
    category_analysis["revenue"].idxmax()
]

most_profitable_category = category_analysis.loc[
    category_analysis["profit"].idxmax()
]

best_customer = top_customers_revenue.iloc[0]


print("\n")
print("=" * 60)
print("              BUSINESS INSIGHTS")
print("=" * 60)

print(
    f"\n1. Highest Revenue Product:"
    f"\n   {best_product['product']}"
    f" | Revenue = {best_product['revenue']:,.2f}"
)

print(
    f"\n2. Most Profitable Product:"
    f"\n   {most_profitable_product['product']}"
    f" | Profit = {most_profitable_product['profit']:,.2f}"
)

print(
    f"\n3. Highest Revenue Category:"
    f"\n   {best_category['category']}"
    f" | Revenue = {best_category['revenue']:,.2f}"
)

print(
    f"\n4. Most Profitable Category:"
    f"\n   {most_profitable_category['category']}"
    f" | Profit = {most_profitable_category['profit']:,.2f}"
)

print(
    f"\n5. Top Customer:"
    f"\n   {best_customer['customer']}"
    f" | Revenue = {best_customer['revenue']:,.2f}"
)


print(
    f"\n7. Top 20% Customers Revenue Contribution:"
    f"\n   {customer_contribution:.2f}%"
)

print("\n" + "=" * 60)
print("             ANALYSIS COMPLETED")
print("=" * 60)


