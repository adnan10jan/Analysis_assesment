import pandas as pd


def generate_car_matrix(df)->pd.DataFrame:
    """
    Creates a DataFrame  for id combinations.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Matrix generated with 'car' values, 
                          where 'id_1' and 'id_2' are used as indices and columns respectively.
    """
    #Write your logic here

    car_matrix = df.pivot(index='id_1', columns='id_2', values='car').fillna(0)

    # Set diagonal values to 0
    for col in car_matrix.columns:
        car_matrix.at[col, col] = 0

    return car_matrix

csv_file_path = 'datasets/dataset-1.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(csv_file_path)

# Use the function
result_matrix = generate_car_matrix(df)

# Print or use the result_matrix as needed
print(result_matrix)


def get_type_count(df)->dict:
    """
    Categorizes 'car' values into types and returns a dictionary of counts.

    Args:
        df (pandas.DataFrame)

    Returns:
        dict: A dictionary with car types as keys and their counts as values.
    """
    # Write your logic here

    df['car_type'] = pd.cut(df['car'], bins=[-float('inf'), 15, 25, float('inf')],
                            labels=['low', 'medium', 'high'], right=False)

    # Count the occurrences of each car type
    type_counts = df['car_type'].value_counts().to_dict()

    return type_counts

# Read the CSV file into a DataFrame
csv_file_path = 'datasets/dataset-1.csv'
df = pd.read_csv(csv_file_path)

# Use the function
result_dict = get_type_count(df)

# Print or use the result_dict as needed
print(result_dict)


def get_bus_indexes(df)->list:
    """
    Returns the indexes where the 'bus' values are greater than twice the mean.

    Args:
        df (pandas.DataFrame)

    Returns:
        list: List of indexes where 'bus' values exceed twice the mean.
    """
    # Write your logic here

    bus_mean = df['bus'].mean()

    # Find the indexes where 'bus' values exceed twice the mean
    bus_indexes = df[df['bus'] > 2 * bus_mean].index.tolist()

    # Sort the indexes in ascending order
    bus_indexes.sort()

    return bus_indexes

# Read the CSV file into a DataFrame
csv_file_path = 'datasets/dataset-1.csv'
df = pd.read_csv(csv_file_path)

# Use the function
result_list = get_bus_indexes(df)

# Print or use the result_list as needed
print(result_list)



def filter_routes(df)->list:
    """
    Filters and returns routes with average 'truck' values greater than 7.

    Args:
        df (pandas.DataFrame)

    Returns:
        list: List of route names with average 'truck' values greater than 7.
    """
    # Write your logic here

    route_means = df.groupby('route')['truck'].mean()

    # Filter routes where the average 'truck' value is greater than 7
    filtered_routes = route_means[route_means > 7].index.tolist()

    # Sort the list of routes in ascending order
    filtered_routes.sort()

    return filtered_routes

# Read the CSV file into a DataFrame
csv_file_path = 'datasets/dataset-1.csv'
df = pd.read_csv(csv_file_path)

# Use the function
result_list = filter_routes(df)

# Print or use the result_list as needed
print(result_list)


def multiply_matrix(matrix)->pd.DataFrame:
    """
    Multiplies matrix values with custom conditions.

    Args:
        matrix (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Modified matrix with values multiplied based on custom conditions.
    """
    # Write your logic here

    modified_matrix = matrix.copy(deep=True)

    # Apply custom conditions to multiply values
    modified_matrix = modified_matrix.applymap(lambda x: x * 0.75 if x > 20 else x * 1.25)

    # Round values to 1 decimal place
    modified_matrix = modified_matrix.round(1)

    return modified_matrix

# Assuming 'result_matrix' is the DataFrame obtained from Question 1
# Replace it with the actual DataFrame if the variable name is different
result_matrix = generate_car_matrix(df)

# Use the function
modified_result_matrix = multiply_matrix(result_matrix)

# Print or use the modified_result_matrix as needed
print(modified_result_matrix)


def time_check(df)->pd.Series:
    """
    Use shared dataset-2 to verify the completeness of the data by checking whether the timestamps for each unique (`id`, `id_2`) pair cover a full 24-hour and 7 days period

    Args:
        df (pandas.DataFrame)

    Returns:
        pd.Series: return a boolean series
    """
    # Write your logic here

    # Check if the necessary columns are present in the DataFrame
    required_columns = ['id', 'id_2', 'startDay', 'startTime', 'endDay', 'endTime']
    if not set(required_columns).issubset(df.columns):
        raise ValueError(f"Error: Required columns {required_columns} not found in the DataFrame.")

    # Combine 'startDay' and 'startTime' to create 'start_timestamp'
    df['start_timestamp'] = pd.to_datetime(df['startDay'] + ' ' + df['startTime'], format='%A %H:%M:%S')

    # Combine 'endDay' and 'endTime' to create 'end_timestamp'
    df['end_timestamp'] = pd.to_datetime(df['endDay'] + ' ' + df['endTime'], format='%A %H:%M:%S')

    # Extract day of the week and hour components from the start_timestamp
    df['day_of_week'] = df['start_timestamp'].dt.dayofweek
    df['hour'] = df['start_timestamp'].dt.hour

    # Create a mask for valid weekdays and hours based on start_timestamp
    valid_mask = (df['day_of_week'].between(0, 6)) & (df['hour'].between(0, 23))

    # Group by ('id', 'id_2') and check if there is any invalid timestamp within each group
    result_series = ~df[valid_mask].groupby(['id', 'id_2']).apply(lambda x: x.duplicated(subset=['day_of_week', 'hour'])).any()

    return result_series

# Read the dataset-2.csv into a DataFrame
csv_file_path = 'datasets/dataset-2.csv'
df_dataset_2 = pd.read_csv(csv_file_path)

# Use the function
result_series = time_check(df_dataset_2)

# Print or use the result_series as needed
print(result_series)