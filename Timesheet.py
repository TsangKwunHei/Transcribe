import pandas as pd
from datetime import datetime
import numpy as np
import re

# Import Google Sheets libraries
import gspread
from google.oauth2.service_account import Credentials

def process_time_tracking(csv_file):
    try:
        # Read the CSV file
        df = pd.read_csv(csv_file, delimiter=',')  # Ensure correct delimiter
    except FileNotFoundError:
        print(f"Error: Could not find the file '{csv_file}'")
        print("Please ensure the file exists in the current directory and the path is correct.")
        return None, None, None
    
    # Standardize column names (strip spaces, lowercase)
    df.columns = df.columns.str.strip().str.lower()
    
    # Convert start date and end date to datetime format
    df['start date'] = pd.to_datetime(df['start date'], errors='coerce')
    df['end date'] = pd.to_datetime(df['end date'], errors='coerce')
    
    # Drop rows where dates could not be converted
    df = df.dropna(subset=['start date', 'end date'])
    
    # Calculate duration in minutes (remove decimals)
    df['calculated duration'] = ((df['end date'] - df['start date']).dt.total_seconds() / 60).astype(int)
    # Also calculate duration in hours (as a float)
    df['duration_hours'] = df['calculated duration'] / 60
    
    # Display processed data
    print("Processed Time Tracking Data:")
    print(df[['timeline', 'start date', 'end date', 'calculated duration', 'duration_hours']])
    
    # Summary: Total time spent per activity (in minutes and hours)
    summary = df.groupby('timeline')['calculated duration'].sum().reset_index()
    summary['duration_hours'] = summary['calculated duration'] / 60
    print("\nTotal Time Spent Per Activity:")
    print(summary)
    
    # Calculate tracking coverage for the week
    # Assume a full week is 7 days: 7 * 24 * 60 = 10080 minutes or 168 hours.
    total_tracked_minutes = df['calculated duration'].sum()
    total_tracked_hours = total_tracked_minutes / 60
    total_possible_minutes = 10080  
    total_possible_hours = total_possible_minutes / 60  # 168 hours
    tracking_percentage = (total_tracked_hours / total_possible_hours) * 100
    
    print(f"\nTracking Coverage: {tracking_percentage:.2f}% of time tracked (week total)")
    
    # Sleep tracking metrics
    sleep_df = df[df['timeline'].str.lower() == 'sleep'].copy()
    
    on_bed_time_variance = None
    avg_sleep_duration = None
    avg_bedtime = None
    avg_wake_time = None
    
    if not sleep_df.empty:
        sleep_df['date'] = sleep_df['start date'].dt.date
        sleep_df['bedtime'] = sleep_df['start date'].dt.strftime('%H:%M:%S')
        sleep_df['wake_time'] = sleep_df['end date'].dt.strftime('%H:%M:%S')
        sleep_df['sleep_duration'] = sleep_df['calculated duration']
        
        bedtime_minutes = sleep_df['start date'].dt.hour * 60 + sleep_df['start date'].dt.minute
        wake_time_minutes = sleep_df['end date'].dt.hour * 60 + sleep_df['end date'].dt.minute
        
        on_bed_time_variance = int(round(np.std(bedtime_minutes)))
        avg_sleep_duration = int(round(sleep_df['sleep_duration'].mean() / 60))
        
        avg_bed_minutes = int(round(bedtime_minutes.mean()))
        avg_bed_hour = avg_bed_minutes // 60
        avg_bed_min = avg_bed_minutes % 60
        avg_bedtime = f"{avg_bed_hour:02d}:{avg_bed_min:02d}"
        
        avg_wake_minutes = int(round(wake_time_minutes.mean()))
        avg_wake_hour = avg_wake_minutes // 60
        avg_wake_min = avg_wake_minutes % 60
        avg_wake_time = f"{avg_wake_hour:02d}:{avg_wake_min:02d}"
        
        print("\nSleep Tracking Metrics:")
        print(f"On-Bed Time Variance: {on_bed_time_variance} min")
        print(f"Avg Sleep Duration: {avg_sleep_duration} h")
        print(f"Avg Bedtime: {avg_bedtime}")
        print(f"Avg Wake-up Time: {avg_wake_time}")
    else:
        print("\nNo sleep data found.")
    
    week_match = re.search(r'week(\d+)', csv_file, re.IGNORECASE)
    if week_match:
        week_number = int(week_match.group(1))
    else:
        week_number = df['start date'].min().isocalendar()[1]
    
    dynamic_categories = ['read', 'learn', 'work']
    dynamic_data = {}
    for cat in dynamic_categories:
        match_row = summary[summary['timeline'].str.lower() == cat]
        if not match_row.empty:
            dynamic_data[cat] = round(match_row['calculated duration'].iloc[0] / 60, 2)
        else:
            dynamic_data[cat] = "Data Missing"
    
    year = df['start date'].min().year
    
    metrics = {
        'year': year,
        'week_number': week_number,
        'total_tracked_hours': total_tracked_hours,
        'tracking_percentage': tracking_percentage,
        'on_bed_time_variance': on_bed_time_variance if on_bed_time_variance is not None else "Data Missing",
        'avg_sleep_duration': avg_sleep_duration if avg_sleep_duration is not None else "Data Missing",
        'avg_bedtime': avg_bedtime if avg_bedtime is not None else "Data Missing",
        'avg_wake_time': avg_wake_time if avg_wake_time is not None else "Data Missing",
        'read': dynamic_data['read'],
        'learn': dynamic_data['learn'],
        'work': dynamic_data['work']
    }
    
    create_weekly_summary_txt(metrics, summary)
    
    return df, summary, metrics

def create_weekly_summary_txt(metrics, summary):
    """
    Creates a TXT file summarizing the week's total tracked time and sleep tracking metrics.
    The file is named 'week{week_number}_summary.txt'.
    """
    filename = f"week{metrics['week_number']}_summary.txt"
    with open(filename, 'w') as f:
        f.write(f"Week {metrics['week_number']} Summary for Year {metrics['year']}\n")
        f.write("=" * 40 + "\n")
        f.write(f"Total Time Tracked: {metrics['total_tracked_hours']:.2f} hours out of 168 hours possible\n")
        f.write(f"Tracking Coverage: {metrics['tracking_percentage']:.2f}%\n\n")
        
        f.write("Sleep Tracking Metrics:\n")
        f.write(f"  On-Bed Time Variance: {metrics['on_bed_time_variance']} min\n")
        f.write(f"  Average Sleep Duration: {metrics['avg_sleep_duration']} h\n")
        f.write(f"  Average Bedtime: {metrics['avg_bedtime']}\n")
        f.write(f"  Average Wake-up Time: {metrics['avg_wake_time']}\n\n")
        
        f.write("Activity Summary (in hours):\n")
        # Sort the summary by duration_hours in descending order and round to one decimal
        sorted_summary = summary.sort_values(by='duration_hours', ascending=False)
        for _, row in sorted_summary.iterrows():
            activity = row['timeline']
            hours = row['duration_hours']
            f.write(f"  {activity.capitalize()}: {hours:.1f} hours\n")
    print(f"Weekly summary TXT file created: {filename}")


if __name__ == "__main__":
    file_path = 'feedback/export_week8_2025_detailed.csv'
    processed_data, summary_data, metrics = process_time_tracking(file_path)
