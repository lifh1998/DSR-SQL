import json
import logging
import re

def quote_field(field_name):
    """为字段名添加引号（如果需要）"""
    field_name = field_name.strip().lower()
    if re.search(r'\W', field_name):
        return f"`{field_name}`"
    else:
        return field_name

def build_database_schema(db_schema_file, related_tables, question_id=None):
    """构建数据库schema"""
    try:
        with open(db_schema_file, 'r', encoding='utf-8') as f:
            schema = json.load(f)
    except FileNotFoundError:
        logging.warning(f"Schema文件不存在: {db_schema_file}")
        return ""
    
    create_statements = []
    for table_name, table_info in schema.items():
        if table_name not in related_tables:
            continue
        
        columns = []
        for column_name, column_info in table_info.items():
            if column_name == "<<key_info>>":
                continue
            
            column_type = column_info.get("type", "")
            column_desc = column_info.get("description", "")
            column_constraints = " ".join(column_info.get("constraints", []))
            column_details = column_info.get("details", "")
            
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
