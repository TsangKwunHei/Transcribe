import pandas as pd
from datetime import datetime
import numpy as np

def process_time_tracking(csv_file):
    # Read the CSV file
    df = pd.read_csv(csv_file, delimiter=',')  # Ensure correct delimiter
    
    # Standardize column names (strip spaces, lowercase)
    df.columns = df.columns.str.strip().str.lower()
    
    # Convert StartDate and EndDate to datetime format
    df['start date'] = pd.to_datetime(df['start date'])
    df['end date'] = pd.to_datetime(df['end date'])
    
    # Calculate duration in minutes (remove decimals)
    df['calculated duration'] = ((df['end date'] - df['start date']).dt.total_seconds() / 60).astype(int)
    
    # Display processed data
    print("Processed Time Tracking Data:")
    print(df[['timeline', 'start date', 'end date', 'calculated duration']])
    
    # Summary: Total time spent per activity
    summary = df.groupby('timeline')['calculated duration'].sum().reset_index()
    print("\nTotal Time Spent Per Activity:")
    print(summary)
    
    # Calculate tracking coverage
    total_tracked_minutes = df['calculated duration'].sum()
    start_date = df['start date'].min()
    end_date = df['end date'].max()
    total_days = (end_date - start_date).days + 1  # Include the first and last day
    total_minutes_possible = total_days * 24 * 60
    tracking_percentage = (total_tracked_minutes / total_minutes_possible) * 100
    
    print(f"\nTracking Coverage: {tracking_percentage:.2f}% of time tracked over {total_days} days")
    
    # Sleep tracking metrics
    sleep_df = df[df['timeline'].str.lower() == 'sleep'].copy()
    
    if not sleep_df.empty:
        sleep_df['date'] = sleep_df['start date'].dt.date
        sleep_df['bedtime'] = sleep_df['start date'].dt.strftime('%H:%M:%S')
        sleep_df['wake_time'] = sleep_df['end date'].dt.strftime('%H:%M:%S')
        sleep_df['sleep_duration'] = sleep_df['calculated duration']
        
        # Extract hours and minutes directly from datetime columns (no need for conversion)
        bedtime_minutes = sleep_df['start date'].dt.hour * 60 + sleep_df['start date'].dt.minute
        wake_time_minutes = sleep_df['end date'].dt.hour * 60 + sleep_df['end date'].dt.minute
        
        # Calculate variances
        sleep_duration_variance = np.std(sleep_df['sleep_duration'])
        bedtime_variance = np.std(bedtime_minutes)
        wake_time_variance = np.std(wake_time_minutes)
        
        # Calculate total sleep duration per day and include sleep begin & wake up time
        daily_sleep = sleep_df.groupby('date').agg(
            total_sleep=('sleep_duration', 'sum'),
            sleep_begin=('bedtime', 'first'),
            wake_up=('wake_time', 'last')
        ).reset_index()
        
        print("\nSleep Tracking Metrics:")
        print(f"Sleep Duration Variance: {sleep_duration_variance:.2f} minutes")
        print(f"Bedtime Variance: {bedtime_variance:.2f} minutes")
        print(f"Wake Time Variance: {wake_time_variance:.2f} minutes")
        
        print("\nTotal Sleep Per Day:")
        print(daily_sleep)
    else:
        print("\nNo sleep data found.")
    
    return df, summary

# Example usage (replace 'csv.csv' with your actual file name)
processed_data, summary_data = process_time_tracking('csv.csv')
