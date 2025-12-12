import os
import json
import pandas as pd
import logging
import re
import itertools
import sqlite3
import argparse
from tqdm import tqdm
from sql_metadata import Parser
from sql_regularizator import format_and_lowercase_sql_query
from pyserini.search.lucene import LuceneSearcher
from value_retriever import build_index_for_dataset, retrieve_relevant_hits, retrieve_question_related_db_values, obtain_n_grams

# --- 配置日志 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_json_dataset(json_dataset):
    # Load the dataset from the provided json_dataset path
    df = pd.read_json(json_dataset)

    # Determine dataset type based on json_dataset path and rename columns accordingly
    if "BIRD" in json_dataset:
        if 'SQL' in df.columns:
            df = df.rename(columns={'SQL': 'query'})
    elif "Spider-Syn" in json_dataset:
        df = df.rename(columns={'SpiderSynQuestion': 'question'})

    # 无论数据集类型如何，最终都确保 'query' 列存在，以防万一
    if 'query' not in df.columns:
        logging.warning(f"Dataset {json_dataset} is missing 'query' column after all checks. Adding 'query' column with default 'no sql.' values.")
        df['query'] = "no sql."

    df = df.reset_index(drop=True)
    return df


def try_exec(db_dir, db_id, query):
    """尝试在数据库中执行 SQL 查询，检查其是否有效。"""
    db_path = os.path.join(db_dir, db_id, f"{db_id}.sqlite")
    if not os.path.exists(db_path):
        return False
    try:
        with sqlite3.connect(db_path) as conn:
            conn.execute(query)
        return True
    except Exception:
        return False


def quote_field(field_name):
    field_name = field_name.strip().lower()
    if re.search(r'\W', field_name):
        return f"`{field_name}`"
    else:
        return field_name


def extract_tables(db_schema_dir, db_id, query):
    if query == "no sql.":
        return [] # 如果查询是 "no sql."，则没有表可以提取

    with open(os.path.join(db_schema_dir, f"{db_id}_schema.json"), "r") as f:
        schema = json.load(f)
    try:
        tables = [quote_field(t) for t in Parser(query).tables]
    except ValueError:
        logging.warning(f"Could not parse query: '{query}' for db_id: {db_id}. Returning empty tables.")
        return []
        
    correct_tables = []
    for table in tables:
        if table in schema.keys():
            correct_tables.append(table)
    return correct_tables


def clean_existing_examples(db_schema_dir):
    """
    遍历指定目录下的所有 _schema.json 文件，并移除其中已存在的 'examples' 字段。
    """
    logging.info(f"Cleaning up existing 'examples' fields from schema files in {db_schema_dir}...")
    if not os.path.isdir(db_schema_dir):
        logging.warning(f"Schema directory not found: {db_schema_dir}. Skipping cleanup.")
        return

    for filename in tqdm(os.listdir(db_schema_dir), desc="Cleaning schema files"):
        if filename.endswith("_schema.json"):
            schema_path = os.path.join(db_schema_dir, filename)
            try:
                with open(schema_path, 'r', encoding='utf-8') as f:
                    schema_data = json.load(f)
                
                modified = False
                # 遍历表和列，删除 examples 键
                for table_name, table_content in schema_data.items():
                    if isinstance(table_content, dict):
                        for column_name, column_content in table_content.items():
                            # MODIFICATION: Changed key from 'example' to 'examples'
                            if isinstance(column_content, dict) and 'examples' in column_content:
                                del column_content['examples']
                                modified = True
                
                if modified:
                    with open(schema_path, 'w', encoding='utf-8') as f:
                        json.dump(schema_data, f, indent=4, ensure_ascii=False)
            except (IOError, json.JSONDecodeError) as e:
                logging.error(f"Could not process or clean file {schema_path}: {e}")


def update_schema_files(db_schema_dir, aggregated_examples):
    """
    将聚合在内存中的 example 数据更新到对应的 schema JSON 文件中。
    """
    logging.info("Updating schema files with aggregated example values...")
    if not aggregated_examples:
        logging.warning("`aggregated_examples` is empty. No schema files will be updated.")
        return

    for db_id, db_examples in tqdm(aggregated_examples.items(), desc="Writing examples to schema files"):
        schema_path = os.path.join(db_schema_dir, f"{db_id}_schema.json")
        if not os.path.exists(schema_path):
            logging.warning(f"Schema file not found for db_id: {db_id}. Skipping.")
            continue

        try:
            with open(schema_path, 'r', encoding='utf-8') as f:
                schema_data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logging.error(f"Failed to read schema file {schema_path}: {e}")
            continue

        file_was_modified = False
        for table_name, table_examples in db_examples.items():
            if table_name in schema_data:
                for column_name, column_examples in table_examples.items():
                    if column_name in schema_data[table_name] and isinstance(schema_data[table_name][column_name], dict):
                        # MODIFICATION: The .update() method still works perfectly here.
                        # It will add or overwrite the 'examples' key in the column dictionary.
                        schema_data[table_name][column_name].update(column_examples)
                        file_was_modified = True

        if file_was_modified:
            try:
                with open(schema_path, 'w', encoding='utf-8') as f:
                    json.dump(schema_data, f, indent=4, ensure_ascii=False)
            except IOError as e:
                logging.error(f"Failed to write updated schema to {schema_path}: {e}")


def build_database_schema(db_schema_dir, db_id, question_id, correct_tables=None):
    """
    构建数据库的 schema 字符串表示。
    如果提供了 question_id，则只包含该问题相关的 example。
    """
    schema_path = os.path.join(db_schema_dir, f"{db_id}_schema.json")
    if not os.path.exists(schema_path):
        return ""
    
    with open(schema_path, 'r', encoding='utf-8') as f:
        schema_data = json.load(f)

    create_statements = []
    for table_name, table_info in schema_data.items():
        if correct_tables and table_name not in correct_tables:
            continue
        columns = []
        for column_name, column_info in table_info.items():
            if column_name == "<<key_info>>":
                continue
            column_type = column_info["type"]
            column_desc = column_info["description"]
            column_constraints = " ".join(column_info.get("constraints", []))
            column_details = column_info["details"]
            
            column_def_base = f"{column_name} {column_type} {column_constraints},"
            
            comment_parts = []
            if column_desc or column_details:
                comment_parts.append(f"{column_desc}{column_details}")
            
            example_part = ""
            if 'examples' in column_info and isinstance(column_info['examples'], list):
                found_example = next((ex for ex in column_info['examples'] if ex.get('question_id') == str(question_id)), None)
                if found_example and 'values' in found_example:
                    example_values = found_example['values']
                    example_part = f"Example: {str(example_values)}"
            
            final_comment = ""
            if comment_parts:
                final_comment += f" -- {''.join(comment_parts)}"
            if example_part:
                final_comment += f" {example_part}" if final_comment else f" -- {example_part}" # Add comma if there's already a comment
            
            column_def = f"{column_def_base}{final_comment}"
            columns.append(column_def.strip())

        key_info = table_info.get("<<key_info>>", "")
        col_info = '\n    '.join(columns)
        create_statement = f"""CREATE TABLE {table_name} (
    {col_info}
    {key_info}
);"""
        
        create_statements.append(create_statement)

    return "\n\n".join(create_statements)


def process_dataset(json_dataset, db_dir, db_schema_dir, output_dir, db_content_index_path, use_value_retrieval=False):
    """
    主处理流程：聚合-写入-再读取。
    """
    df = load_json_dataset(json_dataset)

    db_id2searcher = {}
    aggregated_examples = {}

    if use_value_retrieval:
        logging.info(f"Building content index for dataset: {json_dataset} - {db_dir}")
        build_index_for_dataset(
            dataset_name="bird",
            db_path=db_dir,
            save_index_path=db_content_index_path
        )
        logging.info("Loading Lucene searchers for all databases...")
        unique_db_ids = df["db_id"].unique()
        for db_id in tqdm(unique_db_ids, desc="Loading searchers"):
            index_path = os.path.join(db_content_index_path, db_id)
            if os.path.exists(index_path):
                db_id2searcher[db_id] = LuceneSearcher(index_path)

    processed_dataset_temp = []
    error_count = 0
    has_question_id = "question_id" in df.columns

    for index, row in tqdm(df.iterrows(), total=len(df), desc="Phase 1: Aggregating examples"):
        db_id = row["db_id"]
        question_id = row["question_id"] if has_question_id else index
        
        if use_value_retrieval:
            searcher = db_id2searcher.get(db_id)
            if searcher:
                full_question = f"{row['question']}\n{row.get('evidence', '')}".strip()
                search_queries = list(set(obtain_n_grams(full_question, 8) + [full_question]))
                query2hits = retrieve_relevant_hits(searcher, search_queries)
                all_hits = list(itertools.chain.from_iterable(query2hits.values()))
                unique_hits = [dict(t) for t in {tuple(d.items()) for d in all_hits}]
                relevant_values = retrieve_question_related_db_values(unique_hits, full_question)

                # MODIFICATION: Changed 'example' to 'examples' and data structure to a list of objects
                for full_column_name, values in relevant_values.items():
                    try:
                        table_name, column_name = full_column_name.split('.', 1)
                        db_level = aggregated_examples.setdefault(db_id, {})
                        table_level = db_level.setdefault(table_name, {})
                        column_level = table_level.setdefault(column_name, {})
                        
                        # Get or create the list for 'examples'
                        examples_list = column_level.setdefault('examples', [])
                        
                        # Append the new object to the list
                        examples_list.append({
                            "question_id": str(question_id),
                            "values": values
                        })
                    except ValueError:
                        logging.warning(f"Could not parse '{full_column_name}'. Skipping for aggregation.")
        
        query = format_and_lowercase_sql_query(row["query"])
        is_error = not try_exec(db_dir, db_id, query)
        if is_error:
            error_count += 1
        
        processed_dataset_temp.append({
            "question_id": question_id,
            "db_id": db_id,
            "raw_question": row["question"],
            "question": f"{row['question']}\n{row.get('evidence', '')}".strip(),
            "query": query,
            "is_error": is_error,
        })

    if use_value_retrieval:
        update_schema_files(db_schema_dir, aggregated_examples)

    final_dataset = []
    logging.info("Phase 2: Building final dataset with updated schemas...")
    for data_row in tqdm(processed_dataset_temp, desc="Building final dataset rows"):
        db_id = data_row["db_id"]
        query = data_row["query"]
        question_id = data_row["question_id"]
        
        correct_tables = extract_tables(db_schema_dir, db_id, query)
        data_row["database_schema"] = build_database_schema(db_schema_dir, db_id, question_id, None)
        data_row["filtered_database_schema"] = build_database_schema(db_schema_dir, db_id, question_id, correct_tables)
        data_row["correct_tables"] = ", ".join(set(correct_tables))
        final_dataset.append(data_row)

    logging.info(f"Processing complete. {error_count} queries failed to execute.")
    
    df_final = pd.DataFrame(final_dataset)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "processed_dataset.csv")
    df_final.to_csv(output_path, index=False)
    logging.info(f"Final dataset saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Process dataset to include schema with relevant examples.")
    parser.add_argument("--json_dataset", required=True, help="Path to the input JSON dataset (e.g., dev.json).")
    parser.add_argument("--db_dir", required=True, help="Directory containing the database sqlite files.")
    parser.add_argument("--db_schema_dir", required=True, help="Directory containing the _schema.json files.")
    parser.add_argument("--db_content_index_path", required=True, help="Path to save/load the Lucene content index.")
    parser.add_argument("--output_dir", required=True, help="Directory to save the final processed_dataset.csv.")
    parser.add_argument("--use_value_retrieval", action="store_true", help="Whether to perform value retrieval.")
    
    args = parser.parse_args()

    clean_existing_examples(args.db_schema_dir)

    process_dataset(
        json_dataset=args.json_dataset,
        db_dir=args.db_dir,
        db_schema_dir=args.db_schema_dir,
        output_dir=args.output_dir,
        db_content_index_path=args.db_content_index_path,
        use_value_retrieval=args.use_value_retrieval
    )


if __name__ == "__main__":
    main()
