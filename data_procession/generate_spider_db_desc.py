import os
import pandas as pd
import re, sqlite3, json
import logging
from tqdm import tqdm
from collections import defaultdict

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_db_desc(spider_dir, db_dir, output_dir):
    """
    遍历 Spider 数据集下的所有数据库，并生成数据库的 schema JSON 文件。
    由于 Spider 数据集没有表描述的 CSV 文件，列描述将为空。
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 遍历所有数据库目录
    # Spider 数据库通常位于 spider_dir/db_dir/db_name
    db_paths = [d for d in os.listdir(os.path.join(spider_dir, db_dir)) if not d.startswith('.')]
    
    for db_name in tqdm(db_paths, desc="Processing Spider DBs", unit="db"):
        schema_file_path = os.path.join(output_dir, f"{db_name}_schema.json")
        if os.path.exists(schema_file_path):
            logging.info(f"Schema file already exists for {db_name}. Skipping...")
            continue

        db_schema = {}
        db_path = os.path.join(spider_dir, db_dir, db_name, f'{db_name}.sqlite')

        if not os.path.exists(db_path):
            logging.warning(f"Database file not found for {db_name} at {db_path}. Skipping...")
            continue

        conn = None
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # 获取所有表名
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            table_names = [row[0] for row in cursor.fetchall()]

            for table_name in table_names:
                logging.info(f"Processing table: {table_name} in database: {db_name}")
                table_schema = get_table_schema(conn, table_name)
                db_schema[quote_field(table_name)] = table_schema
            
            with open(schema_file_path, "w", encoding='utf-8') as f:
                json.dump(db_schema, f, indent=4, ensure_ascii=False)
            logging.info(f"Generated schema for {db_name} at {schema_file_path}")

        except sqlite3.Error as e:
            logging.error(f"SQLite error processing {db_name}: {e}")
        except Exception as e:
            logging.error(f"An unexpected error occurred processing {db_name}: {e}")
        finally:
            if conn:
                conn.close()

def get_table_schema(conn, table_name):
    """
    从 SQLite 数据库中获取表的 schema 信息，包括列类型、主键、外键等。
    由于 Spider 没有提供列描述，相关字段将为空。
    """
    cursor = conn.cursor()
    
    # 尝试读取表数据以获取空值和重复值信息，以及示例值
    df = None
    try:
        df = pd.read_sql_query(f"SELECT * FROM `{table_name}`", conn)
    except Exception as e: # 捕获更广泛的异常，包括 UnicodeDecodeError
        logging.warning(f"Could not read data from table `{table_name}` due to: {e}. Skipping data-dependent features.")
        df = pd.DataFrame() # 创建一个空DataFrame以避免后续错误

    contains_null = {}
    contains_duplicates = {}

    if not df.empty: # 仅当 DataFrame 不为空时才处理数据相关特性
        contains_null = {
            column: df[column].isnull().any()
            for column in df.columns
        }
        contains_duplicates = {
            column: df[column].duplicated().any()
            for column in df.columns
        }

    # 获取列信息
    cursor.execute(f"PRAGMA table_info(`{table_name}`)")
    columns_info = cursor.fetchall() # (cid, name, type, notnull, dflt_value, pk)

    # 获取主键信息
    primary_keys = [col[1] for col in columns_info if col[5] == 1]
    primary_key_stmt = ", ".join([quote_field(col) for col in primary_keys])
    if primary_key_stmt:
        primary_key_stmt = f"PRIMARY KEY({primary_key_stmt})"
    
    # 获取外键信息
    cursor.execute(f"PRAGMA foreign_key_list(`{table_name}`)")
    foreign_keys = cursor.fetchall() # (id, seq, parent_table, from_column, to_column, on_update, on_delete, match)
    foreign_key_stmt = get_foreign_key_stmt(foreign_keys)

    # 构建表 schema
    table_schema = {}
    for i, column in enumerate(columns_info):
        column_schema = {}
        # 由于 Spider 没有提供列描述，这里留空
        column_description = ""
        details = ""
        constraints = []
        column_name, column_type = column[1], column[2]
        
        tmp_col = column_name.strip()

        # 添加空值和重复值信息，仅当 DataFrame 不为空时
        if not df.empty:
            if not contains_null[tmp_col]: # If it does not include Null, it's NOT NULL
                constraints.append("NOT NULL")
            if not contains_duplicates[tmp_col]: # If it does not include duplicates, it's UNIQUE
                constraints.append("UNIQUE")
        
        column_schema["type"] = column_type
        column_schema["description"] = column_description # 留空
        column_schema["details"] = details
        column_schema["constraints"] = constraints # Store constraints here
        
        # 判断是否是键
        is_key = tmp_col in primary_keys or tmp_col in [fk[3].strip() for fk in foreign_keys]
        column_schema["is_key"] = is_key
        
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
    """
    根据外键信息生成 SQL 风格的外键语句。
    """
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
    """
    根据字段名是否包含特殊字符，决定是否用反引号引用。
    """
    field_name = field_name.strip().lower()
    if re.search(r'\W', field_name):
        return f"`{field_name}`"
    else:
        return field_name

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate database description for Spider dataset.")
    parser.add_argument('--spider_dir', type=str, required=True, help='Root directory for the Spider dataset.')
    parser.add_argument('--db_dir_suffix', type=str, required=True, help='Suffix for the database directory (e.g., "database").')
    parser.add_argument('--output_dir', type=str, required=True, help='The path of output directory for schema files.')
    args = parser.parse_args()
    
    logging.info(f"Starting to generate database descriptions for Spider dataset.")
    logging.info(f"Spider root directory: {args.spider_dir}")
    logging.info(f"Spider database subdirectory: {args.db_dir_suffix}")
    logging.info(f"Output schema directory: {args.output_dir}")

    generate_db_desc(
        spider_dir=args.spider_dir, 
        db_dir=args.db_dir_suffix,
        output_dir=args.output_dir
    )
    logging.info("Finished generating database descriptions for Spider dataset.")
