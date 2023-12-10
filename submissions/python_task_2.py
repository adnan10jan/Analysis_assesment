import pandas as pd
import numpy as np
from datetime import time



def calculate_distance_matrix(df)->pd.DataFrame():
    """
    Calculate a distance matrix based on the dataframe, df.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Distance matrix
    """
    # Write your logic here

    # Check if the necessary columns are present in the DataFrame
    required_columns = ['id_start', 'id_end', 'distance']
    if not set(required_columns).issubset(df.columns):
        raise ValueError(f"Error: Required columns {required_columns} not found in the DataFrame.")

    # Convert 'id_start' and 'id_end' to int64 to handle 7-digit integer values
    df[['id_start', 'id_end']] = df[['id_start', 'id_end']].astype('int64')

    # Create a new DataFrame for the distance matrix
    matrix_df = pd.DataFrame(index=df['id_start'].unique(), columns=df['id_start'].unique(), dtype='float64')

    # Fill diagonal values with 0
    matrix_df = matrix_df.fillna(0)

    # Iterate through the rows of the DataFrame to calculate cumulative distances
    for index, row in df.iterrows():
        try:
            matrix_df.loc[row['id_start'], row['id_end']] += float(row['distance'])
        except KeyError:
            print(f"KeyError: Unable to find id_start={row['id_start']} or id_end={row['id_end']} in the matrix.")

    # Ensure the matrix is symmetric by copying values to the corresponding positions
   # Ensure the matrix is symmetric by copying values to the corresponding positions
    matrix_df = matrix_df + matrix_df.transpose()

    # Replace diagonal values with the original values (avoid doubling them)
    matrix_df = matrix_df.where(np.triu(np.ones(matrix_df.shape), k=1).astype(bool))
    matrix_df = matrix_df.fillna(matrix_df.transpose())

    # Fill NaN values with 0
    matrix_df = matrix_df.fillna(0)
    return matrix_df

# Read the dataset-3.csv into a DataFrame
csv_file_path = 'datasets/dataset-3.csv'
df_dataset_3 = pd.read_csv(csv_file_path)

# Use the function
result_matrix = calculate_distance_matrix(df_dataset_3)

# Print or use the result_matrix as needed
print(result_matrix)


def unroll_distance_matrix(matrix_df: pd.DataFrame)->pd.DataFrame():
    """
    Unroll a distance matrix to a DataFrame in the style of the initial dataset.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Unrolled DataFrame containing columns 'id_start', 'id_end', and 'distance'.
    """
    # Write your logic here
    unrolled_data = []

    for id_start in matrix_df.index:
        for id_end in matrix_df.columns:
            if id_start != id_end and id_start in matrix_df.index and id_end in matrix_df.columns:
                distance = matrix_df.loc[id_start, id_end]
                unrolled_data.append({'id_start': id_start, 'id_end': id_end, 'distance': distance})

    # Create a DataFrame from the unrolled data
    unrolled_df = pd.DataFrame(unrolled_data)

    return unrolled_df

result_matrix = calculate_distance_matrix(df_dataset_3)

# Unroll the distance matrix
unrolled_df = unroll_distance_matrix(result_matrix)

# Print or use the unrolled_df as needed
print(unrolled_df)


def find_ids_within_ten_percentage_threshold(df, reference_id)->pd.DataFrame():
    """
    Find all IDs whose average distance lies within 10% of the average distance of the reference ID.

    Args:
        df (pandas.DataFrame)
        reference_id (int)

    Returns:
        pandas.DataFrame: DataFrame with IDs whose average distance is within the specified percentage threshold
                          of the reference ID's average distance.
    """
    # Write your logic here

    # Filter the DataFrame for rows with the given reference_id in the 'id_start' column
    reference_rows = df[df['id_start'] == reference_id]

    # Check if the reference_id is present in the DataFrame
    if reference_rows.empty:
        raise ValueError(f"Reference ID {reference_id} not found in the DataFrame.")

    # Calculate the average distance for the reference_id
    reference_avg_distance = reference_rows['distance'].mean()

    # Calculate the threshold values (10% of the average distance)
    threshold_floor = reference_avg_distance - 0.1 * reference_avg_distance
    threshold_ceiling = reference_avg_distance + 0.1 * reference_avg_distance

    # Filter the DataFrame for rows where 'distance' is within the threshold values
    result_df = df[(df['distance'] >= threshold_floor) & (df['distance'] <= threshold_ceiling)]

    # Get unique values from the 'id_start' column and sort them
    result_ids = sorted(result_df['id_start'].unique())

    # Create a DataFrame with the result_ids
    result_df = pd.DataFrame({'id_start': result_ids})

    return result_df

reference_id = 1001436
result_within_threshold = find_ids_within_ten_percentage_threshold(df_dataset_3, reference_id)

# Print or use the result_within_threshold as needed
print(result_within_threshold)

    


def calculate_toll_rate(df)->pd.DataFrame():
    """
    Calculate toll rates for each vehicle type based on the unrolled DataFrame.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Wrie your logic here

    rate_coefficients = {'moto': 0.8, 'car': 1.2, 'rv': 1.5, 'bus': 2.2, 'truck': 3.6}

    df['id_start'] = df['id_start'].iloc[0]

    # Calculate toll rates for each vehicle type
    for vehicle_type, rate_coefficient in rate_coefficients.items():
        df[vehicle_type] = df['distance'] * rate_coefficient


    # Rearrange the columns
    columns_order = ['id_start', 'id_end', 'moto', 'car', 'rv', 'bus', 'truck']
    df_result = df[columns_order]

    return df_result

df_with_toll_rates = calculate_toll_rate(df_dataset_3)

# Print or use df_with_toll_rates as needed
print(df_with_toll_rates)


def calculate_time_based_toll_rates(df)->pd.DataFrame():
    """
    Calculate time-based toll rates for different time intervals within a day.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """

    if 'day' in df.columns:
        df['day'] = pd.to_datetime(df['day']).dt.day


    df['start_day'] = df['start_time'] = df['end_day'] = df['end_time'] = None

    weekday_time_ranges = [
        (time(0, 0, 0), time(10, 0, 0)),
        (time(10, 0, 0), time(18, 0, 0)),
        (time(18, 0, 0), time(23, 59, 59))
    ]
    weekend_time_ranges = [(time(0, 0, 0), time(23, 59, 59))]

    weekday_discount_factors = [0.8, 1.2, 0.8]
    weekend_discount_factor = 0.7

    # Create new columns for start_day, start_time, end_day, and end_time
    #df['start_day'] = df['start_time'] = df['end_day'] = df['end_time'] = None

     # Define your discount factors based on the day of the week
    weekday_discount_factors = [0.9, 0.8, 0.8, 0.8, 0.8, 1.0, 1.0]  # Monday to Sunday

    # Loop through each unique (id_start, id_end) pair
    for pair in df[['id_start', 'id_end']].drop_duplicates().itertuples(index=False):
        for day in range(7):
            # Loop through each time range and apply discount factor based on the day
            for start_time, end_time in weekday_time_ranges if day < 5 else weekend_time_ranges:
                discount_factor = weekday_discount_factors[day] if day < 5 else weekend_discount_factor

                # Set conditions for the specific (id_start, id_end, day, time_range)
                conditions = (
                    (df['id_start'] == pair.id_start) &
                    (df['id_end'] == pair.id_end) &
                    (pd.to_datetime(df['start_day']).dt.day == day) &
                    (df['start_time'] >= start_time) &
                    (df['end_time'] <= end_time)
                )

                # Apply the discount factor to vehicle columns based on the conditions
                df.loc[conditions, ['moto', 'car', 'rv', 'bus', 'truck']] *= discount_factor

    # Map numerical day values to string day names
    day_mapping = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
    df['start_day'] = df['end_day'] = pd.to_datetime(df['start_day']).dt.day.map(day_mapping)
    # df['start_day'] = df['day'].map(day_mapping)
    # df['end_day'] = df['start_day']

    df['start_time'] = pd.to_datetime(df['start_time']).dt.time
    df['end_time'] = pd.to_datetime(df['end_time']).dt.time


    # Convert time column to datetime.time type
    # df['start_time'] = df['start_time']
    # df['end_time'] = df['end_time'] + pd.to_timedelta('1s') - pd.to_timedelta('1us')
    # df['start_time'] = pd.to_datetime(df['start_time']).dt.time
    # df['end_time'] = pd.to_datetime(df['end_time']).dt.time

    # Drop the intermediate 'day' and 'time' columns
    #df.drop(['day', 'time'], axis=1, inplace=True)

    return df

# Assuming df is the DataFrame from Question 3
result_df = calculate_time_based_toll_rates(df_dataset_3)
print(result_df)