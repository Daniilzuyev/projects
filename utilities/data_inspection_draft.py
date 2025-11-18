# draft
# def print_dtypes(df):
#     """Pretty-print column names with their dtypes."""
#     for col, dtype in df.dtypes.items():
#         print(f"{col:20} {dtype}")

# def top_correlations(df, target, ascending=False):
#     """Return sorted correlations of numeric columns with the target."""
#     numeric_df = df.select_dtypes(include=['int64', 'float64'])
#     return numeric_df.corr()[target].sort_values(ascending=ascending)

