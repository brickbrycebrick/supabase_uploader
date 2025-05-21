#!/usr/bin/env python
"""
CSV/XLSX to Supabase Uploader

A CLI tool that helps beginners load data from CSV/XLSX files into Supabase.
It can generate SQL scripts for table creation and upload data to Supabase.
"""

import os
import re
import sys
import json
import time
import random
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, date
from typing import Dict, List, Tuple, Optional, Union, Any
from dotenv import load_dotenv
from supabase import create_client, Client

# Load environment variables
load_dotenv()

# Get Supabase credentials from environment variables
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")


def sanitize_name(name: str) -> str:
    """
    Sanitize a column or table name for SQL compatibility.
    
    Args:
        name: The name to sanitize
        
    Returns:
        Sanitized name
    """
    # Handle None or empty strings
    if not name:
        return "unnamed"
    
    # Convert to lowercase
    sanitized = str(name).lower()
    
    # Handle special cases that commonly cause issues
    # Replace '%' with 'percent'
    sanitized = sanitized.replace('%', 'percent')
    
    # Replace '#' with 'number'
    sanitized = sanitized.replace('#', 'number')
    
    # Replace spaces and special characters with underscores
    sanitized = re.sub(r'[^a-z0-9_]', '_', sanitized)
    
    # Replace multiple consecutive underscores with a single one
    sanitized = re.sub(r'_+', '_', sanitized)
    
    # Remove leading and trailing underscores
    sanitized = sanitized.strip('_')
    
    # Ensure name doesn't start with a number
    if sanitized and sanitized[0].isdigit():
        sanitized = 'n_' + sanitized
    
    # Ensure the name isn't empty after sanitization
    if not sanitized:
        return "unnamed"
    
    # PostgreSQL has a 63-character limit for identifiers
    # Truncate long names to avoid errors
    max_length = 63
    if len(sanitized) > max_length:
        # For very long names, keep the beginning and end parts
        # This helps maintain some context about what the column is
        prefix_length = 30
        suffix_length = max_length - prefix_length - 1  # -1 for the underscore
        sanitized = f"{sanitized[:prefix_length]}_{sanitized[-suffix_length:]}"
    
    return sanitized


def sanitize_table_name(file_path: str) -> str:
    """
    Generate a table name from a file path by extracting the filename
    and sanitizing it for SQL compatibility.
    
    Args:
        file_path: Path to the file
        
    Returns:
        A sanitized table name
    """
    # Extract the filename without extension
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    
    # Sanitize the name for SQL
    sanitized = sanitize_name(base_name)
    
    # Ensure the name starts with a letter or underscore (t_ prefix for tables)
    if sanitized and not sanitized[0].isalpha() and sanitized[0] != '_':
        sanitized = f"t_{sanitized}"
        
    return sanitized


def infer_sql_data_type(series: pd.Series) -> str:
    """
    Infer the SQL data type from a pandas Series.
    
    Args:
        series: The pandas Series to infer the type from
        
    Returns:
        The corresponding SQL data type
    """
    dtype = series.dtype
    non_null_values = series.dropna()
    
    # Check if the series is empty or all null
    if non_null_values.empty:
        return "TEXT"
    
    # Check for numeric types
    if pd.api.types.is_integer_dtype(dtype):
        # Check if values fit in INTEGER range
        if non_null_values.min() >= -2147483648 and non_null_values.max() <= 2147483647:
            return "INTEGER"
        return "BIGINT"
    
    if pd.api.types.is_float_dtype(dtype):
        return "DOUBLE PRECISION"
    
    # Check for boolean
    if pd.api.types.is_bool_dtype(dtype):
        return "BOOLEAN"
    
    # Check for datetime
    if pd.api.types.is_datetime64_dtype(dtype):
        return "TIMESTAMP"
    
    # Check for date
    if isinstance(dtype, pd.PeriodDtype):
        return "DATE"
    
    # Try to detect date strings
    if dtype == 'object':
        # Check if values look like dates (YYYY-MM-DD format)
        sample = non_null_values.iloc[0] if len(non_null_values) > 0 else ""
        if isinstance(sample, str):
            # Try to parse as date
            try:
                # Check if all values match date pattern
                date_pattern = r'^\d{4}-\d{2}-\d{2}$'
                if non_null_values.str.match(date_pattern).all():
                    # Try to convert to datetime to validate
                    pd.to_datetime(non_null_values, format='%Y-%m-%d', errors='raise')
                    return "DATE"
            except (ValueError, TypeError):
                pass
    
    # Default to TEXT for everything else
    return "TEXT"


def generate_create_table_sql(df: pd.DataFrame, table_name: str) -> str:
    """
    Generate SQL for creating a table based on DataFrame schema.
    
    Args:
        df: The pandas DataFrame to generate SQL for
        table_name: The name of the table to create
        
    Returns:
        SQL script as a string
    """
    # Start with table drop and creation
    sql = f"-- Drop the table if it exists\nDROP TABLE IF EXISTS \"{table_name}\";"
    
    sql += "\n\n-- Create the table with inferred schema"
    sql += f"\nCREATE TABLE \"{table_name}\" (\n    \"id\" SERIAL PRIMARY KEY,"
    
    # Create a mapping of original column names to sanitized names
    column_mapping = {col: sanitize_name(col) for col in df.columns}
    
    # Log the column name changes for SQL generation
    print(f"Column name mapping for SQL generation of {table_name}:")
    for original, sanitized in column_mapping.items():
        if original != sanitized:
            print(f"  - '{original}' → '{sanitized}'")
    
    # Add columns based on DataFrame dtypes with sanitized names
    for col in df.columns:
        # Sanitize column name
        col_name = column_mapping[col]
        
        # Infer SQL type from pandas dtype
        if pd.api.types.is_integer_dtype(df[col].dtype):
            sql_type = "INTEGER"
        elif pd.api.types.is_float_dtype(df[col].dtype):
            sql_type = "DOUBLE PRECISION"
        elif pd.api.types.is_datetime64_dtype(df[col].dtype):
            sql_type = "TIMESTAMP"
        elif pd.api.types.is_bool_dtype(df[col].dtype):
            sql_type = "BOOLEAN"
        else:
            sql_type = "TEXT"
        
        sql += f"\n    \"{col_name}\" {sql_type},"
    
    # Remove the trailing comma
    sql = sql.rstrip(',')
    
    sql += "\n);"
    
    return sql


def save_sql_script(sql: str, output_path: str) -> None:
    """
    Save the SQL script to a file. Creates the directory if it doesn't exist.
    
    Args:
        sql: The SQL script to save
        output_path: The path to save the SQL script to
    """
    # Ensure the directory exists
    directory = os.path.dirname(output_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
        
    with open(output_path, 'w') as f:
        f.write(sql)
    print(f"SQL script saved to {output_path}")


def upload_to_supabase(df: pd.DataFrame, table_name: str, mode: str = "overwrite") -> Tuple[bool, str]:
    """
    Upload the DataFrame to Supabase with improved error handling and retry logic.
    
    Args:
        df: The pandas DataFrame to upload
        table_name: The name of the table to upload to
        mode: Upload mode - "overwrite" (default) or "insert"
        
    Returns:
        A tuple of (success, message)
    """
    if not SUPABASE_URL or not SUPABASE_KEY:
        return False, "Supabase credentials not found. Please set SUPABASE_URL and SUPABASE_KEY environment variables."
    
    try:
        # Initialize Supabase client
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        
        # Check if the table exists
        try:
            # Try to get the first row to check if table exists
            table_check = supabase.table(table_name).select("*").limit(1).execute()
        except Exception as e:
            if "relation" in str(e) and "does not exist" in str(e):
                return False, f"Table '{table_name}' does not exist in Supabase. Please create it first using the generated SQL script."
            else:
                # Some other error occurred
                return False, f"Error checking table existence: {str(e)}"
        
        # Convert DataFrame to records and handle non-serializable types
        # First, replace NaN values with None for proper JSON serialization
        df = df.replace({np.nan: None})
        
        # Sanitize column names to match the SQL schema
        # Create a mapping of original column names to sanitized names
        column_mapping = {col: sanitize_name(col) for col in df.columns}
        
        # Rename DataFrame columns to use sanitized names
        df = df.rename(columns=column_mapping)
        
        # Log the column name changes for debugging
        print("Column name mapping for consistency:")
        for original, sanitized in column_mapping.items():
            if original != sanitized:
                print(f"  - '{original}' → '{sanitized}'")
        
        # Custom converter for datetime objects
        def convert_to_serializable(obj):
            if isinstance(obj, (datetime, pd.Timestamp)):
                return obj.isoformat()
            elif isinstance(obj, date):
                return obj.isoformat()
            return obj
        
        # Function to process a single DataFrame for serialization
        def process_dataframe_for_serialization(dataframe):
            # Make a copy to avoid modifying the original
            processed_df = dataframe.copy()
            
            # Replace NaN values with None
            processed_df = processed_df.replace({np.nan: None})
            
            # Convert timestamps and dates to ISO format strings
            for col in processed_df.columns:
                # Check for datetime columns
                if pd.api.types.is_datetime64_dtype(processed_df[col].dtype):
                    processed_df[col] = processed_df[col].apply(lambda x: convert_to_serializable(x) if x is not None else None)
                # Also check for object columns that might contain datetime objects
                elif processed_df[col].dtype == 'object':
                    # Sample the column to check for datetime objects
                    sample = processed_df[col].dropna().head(10)
                    if len(sample) > 0 and any(isinstance(x, (datetime, pd.Timestamp, date)) for x in sample):
                        processed_df[col] = processed_df[col].apply(lambda x: convert_to_serializable(x) if x is not None else None)
            
            return processed_df
        
        # Apply the processing function to the DataFrame
        # In the upload_to_supabase function, we should only have a single DataFrame
        # Multi-sheet handling is done at the main function level
        if isinstance(df, pd.DataFrame):
            df = process_dataframe_for_serialization(df)
        else:  # Dictionary of DataFrames - this shouldn't happen here
            print("Warning: Received a dictionary of DataFrames in upload_to_supabase. This is unexpected.")
            # Just in case, process all sheets
            for sheet_name in df:
                df[sheet_name] = process_dataframe_for_serialization(df[sheet_name])
        
        # Convert DataFrame to records (list of dicts)
        # For single DataFrame case
        if isinstance(df, pd.DataFrame):
            records = df.to_dict(orient='records')
        else:
            # This should never happen in the upload_to_supabase function
            # as the multi-sheet case is handled at the main function level
            # But we'll add this as a safeguard
            print("Warning: Received a dictionary of DataFrames instead of a single DataFrame.")
            records = []
        
        # Function to handle batch upload with retries
        def upload_batch(batch_records, retry_count=0, max_retries=3):
            try:
                result = supabase.table(table_name).insert(batch_records).execute()
                return True, None
            except Exception as e:
                error_str = str(e)
                # Check for specific error types
                if "Object of type Timestamp is not JSON serializable" in error_str:
                    # Fix timestamp serialization issues
                    try:
                        # Convert any remaining Timestamp objects to ISO format strings
                        fixed_batch = []
                        for record in batch_records:
                            fixed_record = {}
                            for key, value in record.items():
                                if isinstance(value, (datetime, pd.Timestamp, date)):
                                    fixed_record[key] = value.isoformat()
                                elif hasattr(value, 'dtype') and pd.api.types.is_datetime64_dtype(value.dtype):
                                    # Handle pandas datetime Series/arrays
                                    fixed_record[key] = value.isoformat()
                                else:
                                    fixed_record[key] = value
                            fixed_batch.append(fixed_record)
                        
                        # Try again with the fixed batch
                        result = supabase.table(table_name).insert(fixed_batch).execute()
                        return True, None
                    except Exception as fix_e:
                        if retry_count < max_retries:
                            print(f"Retrying batch after timestamp fix failed. Error: {str(fix_e)}")
                            time.sleep((2 ** retry_count) + random.uniform(0, 1))
                            return upload_batch(batch_records, retry_count + 1, max_retries)
                        return False, f"Failed to fix timestamp serialization: {str(fix_e)}"
                elif "eof in violation of protocol" in error_str and retry_count < max_retries:
                    # Add exponential backoff with jitter
                    wait_time = (2 ** retry_count) + random.uniform(0, 1)
                    print(f"Encountered 'eof in violation of protocol' error. Retrying in {wait_time:.2f} seconds... (Attempt {retry_count + 1}/{max_retries})")
                    
                    time.sleep(wait_time)
                    return upload_batch(batch_records, retry_count + 1, max_retries)
                return False, error_str
        
        # Handle table clearing for overwrite mode
        if mode == "overwrite":
            try:
                print(f"Deleting all existing data from {table_name}...")
                
                # Try multiple approaches to clear the table
                cleared = False
                
                # Approach 1: Use TRUNCATE via RPC
                try:
                    truncate_query = f"TRUNCATE TABLE \"{table_name}\" RESTART IDENTITY;"
                    supabase.rpc('execute_sql', {'query': truncate_query}).execute()
                    cleared = True
                    print("Table cleared using TRUNCATE statement.")
                except Exception as e1:
                    print(f"TRUNCATE approach failed: {str(e1)}")
                    
                    
                    # Approach 2: Use DELETE with id filter
                    if not cleared:
                        try:
                            supabase.table(table_name).delete().filter('id', 'gte', 0).execute()
                            cleared = True
                            print("Table cleared using DELETE with id filter.")
                        except Exception as e2:
                            print(f"DELETE with id filter failed: {str(e2)}")
                            
                            
                            # Approach 3: Try with columns from the DataFrame
                            if not cleared:
                                for col in df.columns:
                                    try:
                                        supabase.table(table_name).delete().filter(col, 'is', None).or_(f'{col}.not.is', None).execute()
                                        cleared = True
                                        print(f"Table cleared using DELETE with {col} filter.")
                                        break
                                    except:
                                        continue
                
                if not cleared:
                    print("Warning: Could not clear the table. Continuing with insert operation...")
            except Exception as e:
                print(f"Warning: Error during table clearing: {str(e)}")
                
                print("Continuing with insert operation...")
        
        # Upload the data in batches with improved error handling
        batch_size = 1000  # Default batch size
        smaller_batch_size = 100  # Smaller batch size for retries
        total_records = len(records)
        uploaded_count = 0
        failed_records = []
        
        print(f"Uploading {total_records} records in batches of {batch_size}...")
        
        # First pass: Upload in large batches
        for i in range(0, total_records, batch_size):
            end_idx = min(i + batch_size, total_records)
            batch = records[i:end_idx]
            current_batch_size = len(batch)
            
            print(f"Processing batch {i//batch_size + 1}/{(total_records + batch_size - 1)//batch_size} ({i}-{end_idx-1})...")
            
            success, error = upload_batch(batch)
            
            if success:
                uploaded_count += current_batch_size
                print(f"Uploaded {uploaded_count}/{total_records} records...")
            else:
                print(f"Failed to upload batch {i//batch_size + 1}. Error: {error}")
                
                # Add to failed records for retry
                failed_records.extend(batch)
        
        # Second pass: Retry failed records in smaller batches
        if failed_records:
            print(f"Retrying {len(failed_records)} failed records in smaller batches of {smaller_batch_size}...")
            
            retry_failed_count = 0
            for i in range(0, len(failed_records), smaller_batch_size):
                end_idx = min(i + smaller_batch_size, len(failed_records))
                batch = failed_records[i:end_idx]
                
                print(f"Retrying batch {i//smaller_batch_size + 1}/{(len(failed_records) + smaller_batch_size - 1)//smaller_batch_size}...")
                
                success, error = upload_batch(batch, max_retries=5)  # More retries for the smaller batches
                
                if success:
                    retry_failed_count += len(batch)
                    uploaded_count += len(batch)
                    print(f"Successfully uploaded {retry_failed_count}/{len(failed_records)} previously failed records")
                    
                else:
                    print(f"Failed to upload retry batch. Error: {error}")
                    
            
            print(f"Retry process completed. {retry_failed_count}/{len(failed_records)} previously failed records were uploaded.")
        
        # Calculate success rate and prepare final message
        success_rate = (uploaded_count / total_records) * 100 if total_records > 0 else 0
        
        operation_map = {
            "insert": "inserted",
            "overwrite": "overwritten and inserted"
        }
        operation = operation_map.get(mode, "uploaded")
        
        if uploaded_count == total_records:
            return True, f"Successfully {operation} {uploaded_count}/{total_records} records (100%) to {table_name}"
        
        elif uploaded_count > 0:
            return True, f"Partially {operation} {uploaded_count}/{total_records} records ({success_rate:.1f}%) to {table_name}"
        
        else:
            return False, f"Failed to upload any records to {table_name}"
    
    except Exception as e:
        error_msg = str(e)
        if "eof in violation of protocol" in error_msg:
            return False, f"Connection error with Supabase (eof in violation of protocol). Try reducing the batch size or retry later."
        return False, f"Error uploading to Supabase: {error_msg}"


def get_excel_sheet_names(file_path: str) -> Tuple[Optional[List[str]], Optional[str]]:
    """
    Get only the sheet names from an Excel file without loading any data.
    
    Args:
        file_path: Path to the Excel file
        
    Returns:
        A tuple of (list of sheet names, error_message)
    """
    try:
        file_extension = Path(file_path).suffix.lower()
        
        if file_extension not in ['.xlsx', '.xls']:
            return None, f"Not an Excel file: {file_extension}. Please use XLSX or XLS."
            
        # Get sheet names without loading any data
        with pd.ExcelFile(file_path) as excel_file:
            sheet_names = excel_file.sheet_names
            return sheet_names, None
            
    except Exception as e:
        return None, f"Error reading Excel file: {str(e)}"


def read_file(file_path: str, sheet_name: Optional[str] = None, headers_only: bool = False) -> Tuple[Optional[Union[pd.DataFrame, Dict[str, pd.DataFrame]]], Optional[str], Optional[List[str]]]:
    """
    Read a CSV or XLSX file into a pandas DataFrame or dict of DataFrames for multi-sheet Excel files.
    
    Args:
        file_path: Path to the CSV or XLSX file
        sheet_name: Specific sheet name to read (for Excel files only). If None, reads all sheets.
        headers_only: If True, only reads the column headers without loading the entire dataset.
                     This is useful for large files when you only need the schema.
        
    Returns:
        A tuple of (DataFrame or dict of DataFrames, error_message, list of sheet names)
    """
    try:
        file_extension = Path(file_path).suffix.lower()
        available_sheets = None
        
        if file_extension == '.csv':
            if headers_only:
                # Read just a few rows to get the headers, then keep only the column names
                df = pd.read_csv(file_path, nrows=1)
                # Create an empty DataFrame with just the column structure
                df = pd.DataFrame(columns=df.columns)
            else:
                df = pd.read_csv(file_path)
            
        elif file_extension in ['.xlsx', '.xls']:
            # Get available sheet names first
            try:
                with pd.ExcelFile(file_path) as excel_file:
                    available_sheets = excel_file.sheet_names
                    
                    # If a specific sheet is requested
                    if sheet_name is not None:
                        # Make sheet name comparison case-insensitive
                        sheet_name_lower = sheet_name.lower()
                        available_sheets_lower = [s.lower() for s in available_sheets]
                        
                        if sheet_name_lower in available_sheets_lower:
                            # Find the actual sheet name with correct case
                            actual_sheet_name = available_sheets[available_sheets_lower.index(sheet_name_lower)]
                            # Parse dates when reading Excel files
                            if headers_only:
                                # Read just a few rows to get the headers
                                df = pd.read_excel(file_path, sheet_name=actual_sheet_name, nrows=1, parse_dates=True)
                                # Create an empty DataFrame with just the column structure
                                df = pd.DataFrame(columns=df.columns)
                                print(f"Reading headers from sheet: {actual_sheet_name}")
                            else:
                                df = pd.read_excel(file_path, sheet_name=actual_sheet_name, parse_dates=True)
                                print(f"Reading sheet: {actual_sheet_name}")
                            # Store the actual sheet name for later use
                            sheet_name = actual_sheet_name
                        else:
                            return None, f"Sheet '{sheet_name}' not found in Excel file. Available sheets: {', '.join(available_sheets)}", available_sheets
                    else:
                        # Read all sheets into a dictionary of DataFrames
                        if headers_only:
                            # Create a dictionary to store DataFrames with just headers for each sheet
                            df = {}
                            for sheet in available_sheets:
                                # Read just a few rows to get the headers
                                temp_df = pd.read_excel(file_path, sheet_name=sheet, nrows=1, parse_dates=True)
                                # Create an empty DataFrame with just the column structure
                                df[sheet] = pd.DataFrame(columns=temp_df.columns)
                            print(f"Reading headers from all sheets: {', '.join(available_sheets)}")
                        else:
                            df = pd.read_excel(file_path, sheet_name=None, parse_dates=True)
                            print(f"Reading all sheets: {', '.join(available_sheets)}")
                            
            except Exception as excel_error:
                return None, f"Error reading Excel file: {str(excel_error)}", None
        else:
            return None, f"Unsupported file format: {file_extension}. Please use CSV or XLSX.", None
        
        # Check if the DataFrame or all DataFrames in the dict are empty, but skip this check in headers_only mode
        if not headers_only:
            if isinstance(df, pd.DataFrame):
                if df.empty:
                    return None, "The file contains no data.", available_sheets
            else:  # Dictionary of DataFrames
                if all(df_sheet.empty for df_sheet in df.values()):
                    return None, "All sheets in the Excel file are empty.", available_sheets
        
        return df, None, available_sheets
    
    except Exception as e:
        return None, f"Error reading file: {str(e)}", None


def generate_all_tables_sql(dfs: Dict[str, pd.DataFrame], base_table_name: str) -> str:
    """
    Generate SQL to create tables for all sheets in an Excel file.
    
    Args:
        dfs: Dictionary of DataFrames, one per sheet
        base_table_name: Base name for the tables
        
    Returns:
        SQL statement to create all tables
    """
    all_sql = []
    
    for sheet_name, df in dfs.items():
        # Create a table name by combining the base name and sheet name
        sheet_table_name = f"{base_table_name}_{sanitize_name(sheet_name)}"
        
        # Generate SQL for this sheet
        sheet_sql = generate_create_table_sql(df, sheet_table_name)
        
        all_sql.append(sheet_sql)
    
    # Combine all SQL statements
    combined_sql = "\n\n".join(all_sql)
    
    # Add a header comment
    header = f"-- SQL script generated by CSV/XLSX to Supabase Uploader\n-- Combined script for all sheets in Excel file\n"
    
    return header + combined_sql


def main():
    """Main function to run the CLI tool."""
    parser = argparse.ArgumentParser(
        description="Upload CSV/XLSX files to Supabase and generate SQL scripts."
    )
    
    parser.add_argument(
        "file_path",
        help="Path to the CSV or XLSX file"
    )
    
    parser.add_argument(
        "--generate-sql",
        action="store_true",
        help="Generate SQL script for table creation"
    )
    
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload data to Supabase"
    )
    
    parser.add_argument(
        "--sql-output",
        help="Path to save the generated SQL script (defaults to sql_scripts/<table_name>.sql)"
    )
    
    parser.add_argument(
        "--insert-only",
        action="store_true",
        help="Use insert instead of overwrite (default is overwrite which replaces all existing data)"
    )
    
    parser.add_argument(
        "--sheet",
        help="Specific sheet name to process (for Excel files with multiple sheets)"
    )
    
    parser.add_argument(
        "--list-sheets",
        action="store_true",
        help="List all available sheets in an Excel file and exit"
    )
    
    args = parser.parse_args()
    
    # Just list sheets if requested
    if args.list_sheets:
        # Use the optimized function to get only sheet names without loading data
        available_sheets, error = get_excel_sheet_names(args.file_path)
        if error:
            print(f"Error: {error}")
            sys.exit(1)
        
        if available_sheets:
            print(f"Available sheets in {args.file_path}:")
            for i, sheet in enumerate(available_sheets, 1):
                print(f"  {i}. {sheet}")
        else:
            print(f"No sheets found or not an Excel file: {args.file_path}")
        
        sys.exit(0)
    
    # Determine if we only need headers (for SQL generation without upload)
    headers_only = args.generate_sql and not args.upload
    
    # Read the file with optional sheet specification
    df, error, available_sheets = read_file(args.file_path, args.sheet, headers_only=headers_only)
    if error:
        print(f"Error: {error}")
        sys.exit(1)
        
    # If we need the full data for upload but only loaded headers, reload with full data
    if headers_only and args.upload:
        print("Loading full dataset for upload...")
        df, error, _ = read_file(args.file_path, args.sheet, headers_only=False)
        if error:
            print(f"Error loading full dataset: {error}")
            sys.exit(1)
    
    # Get base table name from filename
    base_table_name = sanitize_table_name(args.file_path)
    
    # For single sheet specification, adjust the table name to include the sheet name
    if args.sheet:
        # If user specified a sheet, append sheet name to the base table name
        base_table_name = f"{base_table_name}_{sanitize_name(args.sheet)}"
    
    # Handle multi-sheet Excel files
    is_multi_sheet = isinstance(df, dict)
    
    # Generate SQL if requested
    if args.generate_sql:
        if is_multi_sheet:
            # Generate SQL for all sheets
            print(f"Generating SQL for {len(df)} sheets using base table name: {base_table_name}")
            
            # Determine output path for SQL
            if args.sql_output:
                sql_output = args.sql_output
            else:
                sql_output = os.path.join("sql_scripts", f"{base_table_name}_all_sheets.sql")
            
            # Generate combined SQL for all sheets
            sql = generate_all_tables_sql(df, base_table_name)
            save_sql_script(sql, sql_output)
            
            # Also generate individual SQL files for each sheet
            for sheet_name, sheet_df in df.items():
                sheet_table_name = f"{base_table_name}_{sanitize_name(sheet_name)}"
                sheet_sql = generate_create_table_sql(sheet_df, sheet_table_name)
                sheet_sql_output = os.path.join("sql_scripts", f"{sheet_table_name}.sql")
                save_sql_script(sheet_sql, sheet_sql_output)
                print(f"Generated SQL for sheet '{sheet_name}' as {sheet_sql_output}")
                
        else:
            # Single sheet or CSV file
            print(f"Using table name: {base_table_name}")
            sql = generate_create_table_sql(df, base_table_name)
            
            # Determine output path for SQL (default to sql_scripts folder)
            if args.sql_output:
                sql_output = args.sql_output
            else:
                sql_output = os.path.join("sql_scripts", f"{base_table_name}.sql")
                
            save_sql_script(sql, sql_output)
    
    # Upload to Supabase if requested
    if args.upload:
        if not args.generate_sql:
            print("Warning: Uploading without generating SQL first. Make sure the table exists in Supabase.")
        
        # Determine upload mode based on flags
        mode = "insert" if args.insert_only else "overwrite"
        
        if is_multi_sheet:
            # Upload each sheet to a separate table
            print(f"Uploading data from {len(df)} sheets to separate tables...")
            
            for sheet_name, sheet_df in df.items():
                # If user specified a sheet
                if args.sheet:
                    # We've already adjusted base_table_name to include the sheet name
                    sheet_table_name = base_table_name
                else:
                    # Otherwise, create a table name for each sheet
                    sheet_table_name = f"{base_table_name}_{sanitize_name(sheet_name)}"
                    
                print(f"\nProcessing sheet '{sheet_name}' to table '{sheet_table_name}'...")
                print(f"Using {mode} operation for data upload...")
                
                # Pre-process the DataFrame to handle timestamp serialization
                processed_df = sheet_df.copy()
                
                # Replace NaN values with None
                processed_df = processed_df.replace({np.nan: None})
                
                # Sanitize column names to ensure consistency with SQL schema
                # Create a mapping of original column names to sanitized names
                column_mapping = {col: sanitize_name(col) for col in processed_df.columns}
                
                # Rename DataFrame columns to use sanitized names
                processed_df = processed_df.rename(columns=column_mapping)
                
                # Log the column name changes for debugging
                print("Column name mapping for consistency:")
                for original, sanitized in column_mapping.items():
                    if original != sanitized:
                        print(f"  - '{original}' → '{sanitized}'")
                
                # Convert timestamps to ISO format strings
                for col in processed_df.columns:
                    if pd.api.types.is_datetime64_dtype(processed_df[col].dtype):
                        processed_df[col] = processed_df[col].apply(lambda x: x.isoformat() if isinstance(x, (datetime, pd.Timestamp, date)) and pd.notna(x) else None)
                    # Also check for object columns that might contain datetime objects
                    elif processed_df[col].dtype == 'object':
                        # Check if column contains datetime objects
                        sample = processed_df[col].dropna().head(10)
                        if len(sample) > 0 and any(isinstance(x, (datetime, pd.Timestamp, date)) for x in sample if x is not None):
                            processed_df[col] = processed_df[col].apply(lambda x: x.isoformat() if isinstance(x, (datetime, pd.Timestamp, date)) and pd.notna(x) else x)
                
                success, message = upload_to_supabase(processed_df, sheet_table_name, mode=mode)
                print(message)
                
                if not success:
                    print(f"Failed to upload sheet '{sheet_name}'. Continuing with other sheets...")
        else:
            # Single sheet or CSV file
            print(f"Using {mode} operation for data upload...")
            
            # If this is a single sheet from an Excel file (specified with --sheet)
            # The table name should already be correctly set in base_table_name
            table_name = base_table_name
            print(f"Uploading to table: {table_name}")
            
            success, message = upload_to_supabase(df, table_name, mode=mode)
            print(message)
            
            if not success:
                sys.exit(1)
    
    # If neither option was selected, show help
    if not args.generate_sql and not args.upload and not args.list_sheets:
        parser.print_help()
        print("\nError: Please specify at least one action (--generate-sql, --upload, or --list-sheets)")
        sys.exit(1)


if __name__ == "__main__":
    main()
