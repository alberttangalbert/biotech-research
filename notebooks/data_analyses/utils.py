import pandas as pd 
import numpy as np 
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

#check for nulls and infinites
def null_inf_check(df):
    # Initialize an empty list to store the results
    result = []
    # Iterate over each column in the DataFrame
    for col in df.columns:
        # Check if the column is numeric
        if pd.api.types.is_numeric_dtype(df[col]):
            # Count null (NA) values
            null_count = df[col].isnull().sum()
            # Count infinite values (both +inf and -inf)
            inf_count = np.isinf(df[col]).sum()
            
            # Append the result as a dictionary
            result.append({
                'column': col,
                'nulls': null_count,
                'infinites': inf_count
            })
    # Convert the result list into a DataFrame
    result_df = pd.DataFrame(result)
    return result_df

def fill_na_by_date(df, columns, date_column = 'month_end', stat_function='mean'):

    # Define a dictionary of available statistical functions
    stat_functions = {
        'mean': lambda group: group.mean(),
        'median': lambda group: group.median(),
        'max': lambda group: group.max(),
        'min': lambda group: group.min(),
        'sum': lambda group: group.sum(),
        'std': lambda group: group.std()
    }

    # Validate the stat_function
    if stat_function not in stat_functions:
        raise ValueError(f"Unsupported stat_function: {stat_function}. Choose from {list(stat_functions.keys())}")

    # If columns is a string (single column name), convert it to a list
    if isinstance(columns, str):
        columns = [columns]

    # Group by the date column
    def fill_group(group):
        # Fill NaN values in specified columns
        for col in columns:
            if col in group.columns:
                group[col] = group[col].fillna(stat_functions[stat_function](group[col]))
        return group

    # Apply the fill_group function to each group
    df_filled = df.groupby(date_column).apply(fill_group)

    # Reset the index if needed (optional depending on your DataFrame structure)
    df_filled.reset_index(drop=True, inplace=True)

    return df_filled


#ntiles by month
def ntile_by_month(df, month, field, ntiles, column_name):
    ntile_labels = [f'{field}_{i}' for i in range(1, ntiles + 1)]

    #rank and break the tie by choosing the row that comes first. This assigns a unique rank to every row.
    df['rank'] = df[field].rank(method = 'first', ascending = True)

    df[column_name] = df.groupby(month)['rank'].transform(
        lambda x: pd.qcut(x, ntiles, labels=ntile_labels, duplicates='drop')
    )
    
    # Convert the new column to a categorical type
    df[column_name] = pd.Categorical(df[column_name], categories=ntile_labels, ordered=True)
    
    df = df.drop('rank', axis =1)

    return df.reset_index(drop=True)

def z_score_by_date(df, fields, date_field, suffix = '_z', clip=3, fillna_zero = False):
    
    #If a single field is entered as a string, convert to list
    if isinstance(fields, str):
        fields = [fields]

    #Create z-score function to apply to each group
    def calculate_z_score(group):
        for field in fields:
            #Calculate z-score and name as field & suffix
            group[f"{field}{suffix}"] = (group[field] - group[field].mean()) / group[field].std()
            
            #Clip the z-scores to the range (-clip, clip)
            group[f"{field}{suffix}"] = group[f"{field}{suffix}"].clip(lower=-clip, upper=clip)
            
            #Handle missing values if fillna_zero is True
            if fillna_zero:
                group[f"{field}{suffix}"] = group[f"{field}{suffix}"].fillna(0)
        
        return group

    # Step 2: Apply the function to each group. 
    df = df.groupby(date_field, group_keys=False).apply(calculate_z_score)

    return df

#This z-scores data using 0 as the mean. It is meant for strictly positive data where 0 means worst or missing (so bad if missing)
#set fillna_zero to True if want missing data to be 0

def z_score_right_by_date(df, fields, date_field, suffix = '_z', clip=3, fillna_zero = False):
    
    #If a single field is entered as a string, convert to list
    if isinstance(fields, str):
        fields = [fields]

    #Create z-score function to apply to each group
    def calculate_z_score(group):
        for field in fields:
            #Calculate z-score and name as field & suffix
            group[f"{field}{suffix}"] = (group[field] - 0) / group[field].std()
            
            #Clip the z-scores to the range (-clip, clip)
            group[f"{field}{suffix}"] = group[f"{field}{suffix}"].clip(upper=clip)
            
            #Handle missing values if fillna_zero is True
            if fillna_zero:
                group[f"{field}{suffix}"] = group[f"{field}{suffix}"].fillna(0)
        
        return group

    # Step 2: Apply the function to each group. 
    df = df.groupby(date_field, group_keys=False).apply(calculate_z_score)

    return df

#custom summary table to include NA count and datatypes
def custom_summary(df):
    # make this better by
    # revenue, gross profit, total assets, number of indications, number of modalities 
    # chart that shows the bottom and top one percent 
    # windsorize before you do z-score 
    # top one percent value that explains outsized returns 
    summary = df.describe(include = 'all').transpose()
    summary['NAs'] = df.isna().sum()
    summary['data type'] = df.dtypes
    numeric_cols = df.select_dtypes(include=['number']).columns
    summary['median'] = df[numeric_cols].median()
    fields = ['count', 'NAs', 'unique', 'freq', 'median', 'mean', 'std', 'min', '25%', '50%', '75%', 'max', 'data type']
    # Filter out fields that do not exist in the summary table
    fields_in_summary = [field for field in fields if field in summary.columns]
    # Select only the fields that are present
    summary = summary[fields_in_summary]
    summary = summary.transpose()
    return summary

def run_regression(df, dependent_variable, independent_variables):
    # Define the independent and dependent variables
    X = df[independent_variables]
    y = df[dependent_variable]
    
    # Add a constant to the independent variables
    X = sm.add_constant(X)
    
    # Fit the model
    model = sm.OLS(y, X).fit()
    
    # Create a DataFrame for coefficients, t-stats, and p-values
    results_df = pd.DataFrame({
        "coefficient": model.params,
        "p_value": model.pvalues,
        "t_stat": model.tvalues,
    })
    
    # Extract R-squared
    r_squared = model.rsquared
    
    return results_df, r_squared

def run_regression_with_dummies(df, dependent_variable, independent_variables, dummy_vars, dummy_drop_dict):
    #Initialize an empty dataframe to hold all dummy variables
    X_dummy = pd.DataFrame()

    #Create dummies
    for var in dummy_vars:
        # Check if there's a category to drop
        if var in dummy_drop_dict:
            dummies = pd.get_dummies(df[var], dtype=int).drop(columns=dummy_drop_dict[var], axis =1)
        else:
            dummies = pd.get_dummies(df[var], drop_first=True, dtype=int)
            
        #Rename the dummy column to the original column name if it's binary (only one column)
        if len(dummies.columns) == 1:
            dummies.columns = [var]            
            
        X_dummy = pd.concat([X_dummy, dummies], axis=1)

    #define the dependent and independent variables
    X = pd.concat([X_dummy, df[independent_variables]], axis = 1)
    y = df[dependent_variable]
    
    # Add a constant to the independent variables
    X = sm.add_constant(X)
    
    # Fit the model
    model = sm.OLS(y, X).fit()
    
    # Create a DataFrame for coefficients, t-stats, and p-values
    results_df = pd.DataFrame({
        "coefficient": model.params,
        "p_value": model.pvalues,
        "t_stat": model.tvalues,
    })
    
    # Extract R-squared
    r_squared = model.rsquared
    
    return results_df, r_squared

def run_monthly_regression(df, id_variables, dependent_variable, independent_vars, dummy_variables, dummy_drop_dict):
    # Get unique groups
    group_list = df['month_end'].unique()

    # Create empty lists to hold results
    summary_results = []
    regression_results = []
    fitted_data = []

    # Iterate over each asset to run regression
    for group in group_list:
        temp_df = df[df['month_end'] == group].copy()

        # Initialize an empty dataframe to hold all dummy variables
        X_dummy = pd.DataFrame()

        # Create dummies
        for var in dummy_variables:
            # Check if there's a category to drop
            if var in dummy_drop_dict:
                dummies = pd.get_dummies(temp_df[var], dtype=int).drop(columns=dummy_drop_dict[var], axis=1)
            else:
                dummies = pd.get_dummies(temp_df[var], drop_first=True, dtype=int)

            # Rename the dummy column to the original column name if it's binary (only one column)
            if len(dummies.columns) == 1:
                dummies.columns = [var]            

            X_dummy = pd.concat([X_dummy, dummies], axis=1)
        
        
        # Define the dependent and independent variables


        X = pd.concat([X_dummy, temp_df[independent_vars]], axis=1).drop(columns = dummy_variables)
        y = temp_df[dependent_variable]

        # Calculate the mean, standard deviation and MAD of the original dependent variable
        y_mean = y.mean()
        y_std_dev = y.std()
        y_mad = np.mean(np.abs(y - y.mean()))

        # Add a constant to the independent variables
        X = sm.add_constant(X)
        X = X.select_dtypes(exclude=['datetime64[ns]'])

        model = sm.OLS(y, X).fit()

        # Calculate residuals and fitted values
        fitted_values = model.predict(X)
        residuals = y - fitted_values

        # Include id variables in the fitted data
        temp_fitted_data = pd.DataFrame({
            'month_end': group,
            'fitted_value': fitted_values,
            'residual': residuals
        })

        # Add id_variables to the temp_fitted_data DataFrame
        for id_var in id_variables:
            temp_fitted_data[id_var] = temp_df[id_var].values

        fitted_data.append(temp_fitted_data)

        # Calculate RMSE and MAE for the residuals
        rmse = np.sqrt(model.mse_resid)
        mae = residuals.abs().mean()

        # Extract statistics
        results = {
            'month_end': group,
            'r_squared': model.rsquared,
            'y_mean': y_mean,
            'y_stdev': y_std_dev,
            'rmse': rmse,
            'y_mad': y_mad,
            'mae': mae
        }

        # Append results to list
        summary_results.append(results)

        # Add coefficients, including the constant
        for var in X.columns:
            regression_results.append({
                'month_end': group,
                'variable': var,
                'coefficient': model.params[var],
                'p_value': model.pvalues[var],
                't_stat': model.tvalues[var]
            })

    # Create Summary DataFrame
    regression_stats = pd.DataFrame(summary_results)
    regression_stats = regression_stats.sort_values(by='month_end')
    coefficients = pd.DataFrame(regression_results)
    coefficients = coefficients.sort_values(by='month_end')

    coefficients_pivot = coefficients.pivot(
        index='month_end',
        columns='variable',
        values='coefficient'
    )
    coefficients_pivot = coefficients_pivot.reset_index()

    # Combine fitted data
    regression_fitted = pd.concat(fitted_data).reset_index(drop=True)

    return regression_stats, coefficients, coefficients_pivot, regression_fitted

def calc_summary_regression_stats(df): 
    metrics = ['r_squared', 'y_mean', 'y_stdev', 'rmse', 'y_mad', 'mae']

    # Calculate mean, median, and stdev for each metric
    mean_values = df[metrics].mean()
    median_values = df[metrics].median()
    stdev_values = df[metrics].std()

    # Create a dictionary to store these statistics as rows
    statistics_dict = {
        'mean': mean_values,
        'median': median_values,
        'stdev': stdev_values
    }

    # Convert the dictionary into a DataFrame and transpose it
    summary_stats_df = pd.DataFrame(statistics_dict).T
    summary_stats_df = summary_stats_df.reset_index().rename(columns={'index': 'statistic'})
    summary_stats_df['std_improvement'] = summary_stats_df['y_stdev'] - summary_stats_df['rmse'] 
    summary_stats_df['mad_improvement'] = summary_stats_df['y_mad'] - summary_stats_df['mae'] 

    # Display the summary stats dataframe
    return summary_stats_df


def calc_summary_coefficients(df, sort_order = None):

    sort_order = ['const'] + sort_order

    # Calculate summary returns
    summary = df.groupby('variable').agg(
        avg_ceoff = pd.NamedAgg(column= 'coefficient', aggfunc=lambda x: x.mean()),
        stdev = pd.NamedAgg(column= 'coefficient', aggfunc=lambda x: x.std()),
        Sharpe = pd.NamedAgg(column='coefficient', aggfunc=lambda x: abs(x.mean()) / x.std()),
        avg_p_value = pd.NamedAgg(column= 'p_value', aggfunc=lambda x: (x.mean())),
        count_p_value_below_05=pd.NamedAgg(
            column='p_value', 
            aggfunc=lambda x: (x < 0.05).sum()
        ),
        total_month_ends=pd.NamedAgg(
            column='month_end', 
            aggfunc=lambda x: len(x.unique())
        )
        ).reset_index()
    
    summary['perc_below_0.05'] = (
        summary['count_p_value_below_05'] / summary['total_month_ends'])
    summary = summary.drop(columns=['count_p_value_below_05', 'total_month_ends'])

    if sort_order:
        summary['sort_order'] = summary['variable'].apply(
            lambda x: sort_order.index(x) if x in sort_order else float('inf')
        )
        summary = summary.sort_values('sort_order').drop(columns=['sort_order'])

    return summary

def calculate_cumulative_coefficients(coefficients_pivot, month_end_column):
    
    # Ensure the DataFrame is sorted by the month_end_column
    coefficients_pivot = coefficients_pivot.sort_values(by=month_end_column)

    # Compute the cumulative sum for each numeric column (excluding the ordering column)
    cumulative_df = coefficients_pivot.drop(columns=[month_end_column]).cumsum()

    # Reattach the ordering column
    cumulative_df.insert(0, month_end_column, coefficients_pivot[month_end_column])

    return cumulative_df


def correlation_table(df, field_list): 
    df = df[field_list].copy()
    correlation_matrix = df.corr()
    plt.figure(figsize=(20, 16))
    ax = sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    # Move the x-axis labels to the top
    ax.xaxis.set_ticks_position('top')
    plt.xticks(rotation=90)
    plt.title('Correlation Matrix Heatmap')
    plt.show()

def time_series_df_plot(df, date, field_list, width=12, height=6):
    # Create figure
    plt.figure(figsize=(width, height))
    
    # Calculate number of lines we'll actually plot (excluding date column)
    num_lines = len([col for col in field_list if col != date])
    
    # Only generate colors if we have lines to plot
    if num_lines > 0:
        colors = plt.cm.tab20(np.linspace(0, 1, num_lines))
    
    # Ensure the month_end_column is the x-axis
    x_values = df[date]
    
    # Plot each numeric column as a line with a distinct color
    color_index = 0
    for col in field_list:
        if col != date:
            plt.plot(x_values, df[col], label=col, color=colors[color_index])
            color_index += 1
    
    # Add chart details
    plt.xlabel(date)
    plt.xticks(rotation=45)
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.grid(True)
    
    # Show the plot
    plt.tight_layout()
    plt.show()

#create corr pairs tuple list. Can't use dict because named by key 
corr_pairs_dict = [
    ('ytw', 'const'),
    ('dtw', 'const'),
    ('ytw', 'dtw'),
]

def rolling_correlations_df(df, date, corr_pairs_list, window):
    # Create an empty DataFrame with dates
    results = df.reset_index()[[date]].copy()
    for key, value in corr_pairs_list:
        column_name = f'{key}_vs_{value}'
        results[column_name] = df[key].rolling(window=window, min_periods=3).corr(df[value])
    return results

def plot_distribution(df, columns):  
    # Ensure that 'fields' is a list, even if a single field is passed
    if isinstance(columns, str):
        columns = [columns]
    for column in columns:
        plt.figure(figsize=(8, 6))
        sns.histplot(df[column], bins=30, kde=True, edgecolor='black', alpha=0.7)
        plt.title(f'Distribution of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.show()