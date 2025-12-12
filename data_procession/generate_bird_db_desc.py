import os
import pandas as pd
import re, sqlite3, os, chardet
import logging
import json
from tqdm import tqdm
from collections import defaultdict

def generate_db_desc(bird_dir, db_dir, output_dir):
    """
    Traverse all directories under bird_dir/db_dir and read the .csv files
    in the database_description folder using pandas.
    """
    # Traverse all directories under bird_dir/db_dir
    subdirs = [d for d in os.listdir(os.path.join(bird_dir, db_dir)) if not d.startswith('.')]
    for subdir in tqdm(subdirs, desc="Processing", unit="dir"):
        # print("db_name:", subdir)
        schema_file_path = os.path.join(output_dir, f"{subdir}_schema.json")
        if os.path.exists(schema_file_path):
            logging.info(f"Schema file already exists for {subdir}. Skipping...")
            continue

        db_schema = {}
        db_desc_dir = os.path.join(bird_dir, db_dir, subdir, 'database_description')
        conn = sqlite3.connect(os.path.join(bird_dir, db_dir, subdir, f'{subdir}.sqlite'))
        for filename in os.listdir(db_desc_dir):
            if not filename.endswith('.csv'):
                continue
            table_name = filename[: -4]
            logging.info(f"Processing file: {filename}")
            csv_path = os.path.join(db_desc_dir, filename)
            with open(csv_path, 'rb') as f:
                result = chardet.detect(f.read())
            table_df = pd.read_csv(csv_path, encoding=result['encoding'])
            
            table_schema = get_table_schema(conn, table_name, table_df)
            db_schema[quote_field(table_name)] = table_schema
        with open(schema_file_path, "w") as f:
            json.dump(db_schema, f, indent=4)
        conn.close()
                
def get_table_schema(conn, table_name, table_df):
    # print("table_name:", table_name)
    cursor = conn.cursor()
    df = pd.read_sql_query(f"SELECT * FROM `{table_name}`", conn)
    contains_null = {
        column: df[column].isnull().any()
        for column in df.columns
    }
    contains_duplicates = {
        column: df[column].duplicated().any()
        for column in df.columns
    }
    dic = {}
    for _, row in table_df.iterrows():
        try:
            if not row.iloc[0]:
                continue
            col_description, val_description = "", ""
            col = row.iloc[0].strip()
            if pd.notna(row.iloc[2]):
                col_description = re.sub(r'\s+', ' ', str(row.iloc[2]))
            if col_description.strip() == col or col_description.strip() == "":
                col_description = ''
            if pd.notna(row.iloc[4]):
                val_description = re.sub(r'\s+', ' ', str(row.iloc[4]))
            if val_description.strip() == "" or val_description.strip() == col or val_description == col_description:
                val_description = ""
            col_description = col_description[:200]
            val_description = val_description[:200]
            dic[col] = col_description, val_description
        except Exception as e:
            print(e)
            dic[col] = "", ""

    # Get columns info
    cursor.execute(f"PRAGMA table_info(`{table_name}`)")
    columns_info = cursor.fetchall()
    # Get primary key info
    primary_keys = [col[1] for col in columns_info if col[5] == 1]
    primary_key_stmt = ", ".join([quote_field(col) for col in primary_keys])
    if primary_key_stmt:
        primary_key_stmt = f"PRIMARY KEY({primary_key_stmt})"
    # Get foreign key info
    cursor.execute(f"PRAGMA foreign_key_list(`{table_name}`)")
    foreign_keys = cursor.fetchall()
    foreign_key_stmt = get_foreign_key_stmt(foreign_keys)
    # Build table schema
    table_schema = {}
    for column in columns_info: # Removed 'val' from zip
        column_schema = {}
        column_description = ""
        details = ""
        constraints = []
        column_name, column_type, not_null, default_value, pk = column[1:6]
        # Add type and description information
        tmp_col = column_name.strip()
        col_des, val_des = dic.get(tmp_col, ["", ""])
        col_des.strip()
        val_des.strip()
        if col_des != "":
            column_description += col_des if col_des.endswith('.') else f"{col_des}. "
        if val_des != "":
            details +=  val_des if val_des.endswith('.') else f"{val_des}. "
        
        if not contains_null[tmp_col]: # If it does not include Null, it's NOT NULL
            constraints.append("NOT NULL")
        if not contains_duplicates[tmp_col]: # If it does not include duplicates, it's UNIQUE
            constraints.append("UNIQUE")

        # Removed sample value details from here
        column_schema["type"] = column_type
        column_schema["description"] = column_description
        column_schema["details"] = details
        column_schema["constraints"] = constraints # Store constraints here
        # Add additional information
        column_schema["is_key"] = tmp_col in primary_keys or tmp_col in [col.strip() for _, _, _, col, _, _, _, _ in foreign_keys]
        
        table_schema[quote_field(tmp_col)] = column_schema
    
    key_info = []
    if primary_key_stmt:
        key_info.append(primary_key_stmt)
    if foreign_key_stmt:
        key_info.append(foreign_key_stmt)
    table_schema["<<key_info>>"] = ",\n    ".join(key_info)
    cursor.close()

    return table_schema

def get_foreign_key_stmt(foreign_keys):
    foreign_key_groups = defaultdict(list)
    for foreign_key in foreign_keys:
        foreign_key_groups[foreign_key[0]].append(foreign_key)
    foreign_key_stmts = []

    for group in foreign_key_groups.values():
        group = sorted(group, key=lambda fk: fk[1])  # seq = fk[1]
        columns = [quote_field(fk[3].strip()) for fk in group]  # from = fk[3]
        ref_table = quote_field(group[0][2])           # table = fk[2]
        ref_columns = [
            quote_field(fk[4].strip()) if fk[4] is not None else None
            for fk in group
        ]
        if None in ref_columns: 
            foreign_key_stmt = (
                f"FOREIGN KEY ({', '.join(columns)}) "
                f"REFERENCES {ref_table}"
            )
        else: 
            foreign_key_stmt = (
                f"FOREIGN KEY ({', '.join(columns)}) "
                f"REFERENCES {ref_table}({', '.join(ref_columns)})"
            )
        foreign_key_stmts.append(foreign_key_stmt)
    return ",\n    ".join(foreign_key_stmts)

def quote_field(field_name):
    field_name = field_name.strip().lower()
    if re.search(r'\W', field_name):
        return f"`{field_name}`"
    else:
        return field_name

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate database description.")
    parser.add_argument('--bird_dir', type=str, required=True, help='Root directory for the BIRD dataset.')
    parser.add_argument('--db_dir_suffix', type=str, required=True, help='Suffix for the database directory (e.g., "dev/dev_databases").')
    parser.add_argument('--output_dir', type=str, required=True, help='The path of output directory for schema files.')
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info(f"Start generating database description, output directory: {args.output_dir}.")
    
    generate_db_desc(
        bird_dir=args.bird_dir, 
        db_dir=args.db_dir_suffix,
        output_dir=args.output_dir
    )
