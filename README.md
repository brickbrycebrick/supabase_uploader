# CSV/XLSX to Supabase Uploader

A CLI tool that helps beginners load data from CSV/XLSX files into Supabase. It can generate SQL scripts for table creation and upload data to Supabase.

## Features

- Supports both CSV and XLSX file formats
- Handles multi-sheet Excel files with options to process specific sheets
- Automatically determines column data types and maps them to PostgreSQL types
- Generates SQL scripts for table creation (saved to `sql_scripts` folder by default)
- Uploads data to Supabase with proper error handling
- Offers two data loading modes: overwrite (default) and insert-only
- Converts file name to a valid table name (removes special characters, replaces spaces with underscores)
- Handles data in batches to avoid timeouts
- Provides clear error messages and progress feedback

## Prerequisites

1. Python 3.7 or higher
2. A Supabase account and project
3. Required Python packages (install via `pip install -r requirements.txt`):
   - pandas
   - supabase
   - python-dotenv
   - openpyxl (for Excel file support)

## Setup

1. Create a `.env` file in the project directory with your Supabase credentials:

```
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
```

2. Install the required dependencies:

```
pip install -r requirements.txt
```

## Usage

The CLI tool offers two main functions:

1. Generate SQL scripts for table creation
2. Upload data to Supabase

### Basic Usage

```bash
# Generate SQL script only
python upload_to_supabase.py path/to/your/file.csv --generate-sql

# Upload data to Supabase only
python upload_to_supabase.py path/to/your/file.xlsx --upload

# Generate SQL script and upload data
python upload_to_supabase.py path/to/your/file.csv --generate-sql --upload
```

### Advanced Options

```bash
# Specify a custom output path for the SQL script
python upload_to_supabase.py path/to/your/file.csv --generate-sql --sql-output path/to/output.sql

# Use insert-only mode instead of overwrite (default is overwrite which replaces all existing data)
python upload_to_supabase.py path/to/your/file.csv --upload --insert-only

# List all available sheets in an Excel file
python upload_to_supabase.py path/to/your/file.xlsx --list-sheets

# Process a specific sheet from an Excel file
python upload_to_supabase.py path/to/your/file.xlsx --generate-sql --upload --sheet "Sheet1"
```

## Notes

- The table name will be derived from the file name (with spaces replaced by underscores and special characters removed)
- For multi-sheet Excel files:
  - By default, all sheets will be processed with table names as `filename_sheetname`
  - Use `--sheet "Sheet Name"` to process only a specific sheet
  - Use `--list-sheets` to see all available sheets in an Excel file
- The generated SQL script will include a `DROP TABLE IF EXISTS` statement to ensure a clean slate
- SQL scripts are saved to the `sql_scripts` folder by default
- When uploading data, make sure the table already exists in Supabase with the correct schema
- The tool supports two data loading modes:
  - **Overwrite** (default): Deletes all existing data before inserting new records
  - **Insert-only**: Only inserts new records (will fail if duplicates exist)
- Use the `--insert-only` flag for insert-only mode
- Data is uploaded in batches to avoid request size limits
- The tool requires appropriate permissions in your Supabase project

## Examples

### Generate SQL Script

```bash
python upload_to_supabase.py data/employees.csv --generate-sql
```

This will generate an `sql_scripts/employees.sql` file with SQL commands to create a table with the appropriate schema.

### Upload Data

```bash
python upload_to_supabase.py data/employees.xlsx --upload
```

### Working with Multi-Sheet Excel Files

```bash
# List all sheets in an Excel file
python upload_to_supabase.py data/quarterly_reports.xlsx --list-sheets

# Generate SQL for a specific sheet
python upload_to_supabase.py data/quarterly_reports.xlsx --generate-sql --sheet "Q1 2023"

# Upload data from a specific sheet
python upload_to_supabase.py data/quarterly_reports.xlsx --upload --sheet "Q1 2023"

# Generate SQL and upload data from a specific sheet
python upload_to_supabase.py data/quarterly_reports.xlsx --generate-sql --upload --sheet "Q1 2023"
```

This will upload the data from the Excel file to a table named `employees` in your Supabase project, replacing any existing data.

### Complete Workflow

```bash
python upload_to_supabase.py data/sales_2023.csv --generate-sql --upload --table-name annual_sales
```

This will:
1. Generate an `sql_scripts/annual_sales.sql` file with table creation commands
2. Upload the data to a table named `annual_sales` in your Supabase project, replacing any existing data
