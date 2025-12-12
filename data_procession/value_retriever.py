import json
import shutil
import sqlite3
import os
from tqdm import tqdm
import re
import random
from collections import OrderedDict
from pyserini.search.lucene import LuceneSearcher
from nltk.tokenize import word_tokenize
from nltk import ngrams
from func_timeout import func_set_timeout, FunctionTimedOut

SQL_RESERVED_WORDS = {'IDENTIFIED', 'FOREIGN', 'CONSTRAINT', 'USER', 'POSITION', 'DESCRIBE', 'CHECK', 'RECURSIVE',
                      'REAL', 'CONTINUE', 'GLOBAL', 'RLIKE', 'INSENSITIVE', 'BOOLEAN', 'CHAR', 'ROLE', 'CASE', 'SCHEMA',
                      'CLOB', 'RESIGNAL', 'ROW', 'DEC', 'TOP', 'EXCEPT', 'SENSITIVE', 'OUT', 'RENAME', 'READS', 'BLOB',
                      'INT', 'EXTERNAL', 'LOCALTIMESTAMP', 'DECLARE', 'DO', 'AS', 'OVER', 'CONDITION', 'SELECT',
                      'SAVEPOINT', 'WITHIN', 'ELSEIF', 'UNLOCK', 'DATABASE', 'TRIGGER', 'ACCESS', 'FALSE', 'BREAK',
                      'ITERATE', 'SMALLINT', 'ASC', 'YEAR', 'DELETE', 'ROLLBACK', 'ON', 'ESCAPE', 'CREATE', 'MONTH',
                      'SPECIFIC', 'SESSION', 'SQLSTATE', 'HOLD', 'SET', 'EXPLAIN', 'RETURN', 'ROWNUM', 'BINARY',
                      'SYSDATE', 'SQLWARNING', 'EXTEND', 'CAST', 'FOR', 'TERMINATED', 'VIEW', 'TRAILING', 'HOUR',
                      'VARYING', 'RESTRICT', 'RIGHT', 'DISTINCT', 'JOIN', 'UNKNOWN', 'VALUES', 'TABLE', 'OR', 'DOUBLE',
                      'DROP', 'COMMIT', 'PRECISION', 'LANGUAGE', 'START', 'INTERSECT', 'IGNORE', 'NULL', 'CURRENT_DATE',
                      'LOCK', 'INTO', 'NEW', 'DESC', 'STATIC', 'MODIFIES', 'GRANT', 'VALUE', 'LIMIT', 'MODULE', 'DATE',
                      'LOCALTIME', 'PERCENT', 'REPEAT', 'FULL', 'USAGE', 'ORDER', 'WHEN', 'PRIMARY', 'BETWEEN',
                      'CURSOR', 'DECIMAL', 'HAVING', 'IF', 'FILTER', 'INDEX', 'ILIKE', 'VARCHAR', 'EXEC', 'USING',
                      'ROWS', 'PLACING', 'WHILE', 'EXECUTE', 'EACH', 'LEFT', 'FLOAT', 'COLLATE', 'CURRENT_TIME', 'OPEN',
                      'RANGE', 'CROSS', 'FUNCTION', 'TIME', 'BOTH', 'NOT', 'CONVERT', 'NCHAR', 'KEY', 'DEFAULT', 'LIKE',
                      'ANALYZE', 'EXISTS', 'IN', 'BIT', 'INOUT', 'SUM', 'NUMERIC', 'AFTER', 'LEAVE', 'INSERT', 'TO',
                      'COUNT', 'THEN', 'BEFORE', 'OUTER', 'COLUMN', 'ONLY', 'END', 'PROCEDURE', 'OFFSET', 'ADD',
                      'INNER', 'RELEASE', 'FROM', 'DAY', 'NO', 'CALL', 'BY', 'LOCAL', 'ZONE', 'TRUE', 'EXIT', 'LEADING',
                      'INTEGER', 'MERGE', 'OLD', 'AVG', 'MIN', 'SQL', 'LOOP', 'SIGNAL', 'REFERENCES', 'MINUTE',
                      'UNIQUE', 'GENERATED', 'ALL', 'MATCH', 'CASCADE', 'UNION', 'COMMENT', 'FETCH', 'UNDO', 'UPDATE',
                      'WHERE', 'ELSE', 'PARTITION', 'BIGINT', 'CHARACTER', 'CURRENT_TIMESTAMP', 'ALTER', 'INTERVAL',
                      'REVOKE', 'CONNECT', 'WITH', 'TIMESTAMP', 'GROUP', 'BEGIN', 'CURRENT', 'REGEXP', 'NATURAL',
                      'SOME', 'SQLEXCEPTION', 'MAX', 'SUBSTRING', 'OF', 'AND', 'REPLACE', 'IS'}
SPECIAL_CHARS_PATTERN = re.compile(r'[^a-zA-Z0-9_]')

def get_cursor_from_path(sqlite_path):
    try:
        if not os.path.exists(sqlite_path):
            print("Open a new connection %s" % sqlite_path)
        connection = sqlite3.connect(sqlite_path, check_same_thread=False)
    except Exception as e:
        print(sqlite_path)
        raise e
    connection.text_factory = lambda b: b.decode(errors="ignore")
    cursor = connection.cursor()
    return cursor

@func_set_timeout(3600)
def execute_sql(cursor, sql):
    cursor.execute(sql)
    return cursor.fetchall()

def remove_contents_of_a_folder(index_path):
    os.makedirs(index_path, exist_ok=True)
    for filename in os.listdir(index_path):
        file_path = os.path.join(index_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def build_content_index(db_file_path: str, index_path: str):
    cursor = get_cursor_from_path(db_file_path)
    results = execute_sql(cursor, "SELECT name FROM sqlite_master WHERE type='table';")
    table_names = [result[0] for result in results]

    all_column_contents = []
    for table_name in table_names:
        if table_name == "sqlite_sequence":
            continue
        results = execute_sql(cursor, f"SELECT name FROM PRAGMA_TABLE_INFO('{table_name}')")
        column_names_in_one_table = [result[0] for result in results]
        for column_name in column_names_in_one_table:
            try:
                print(f"SELECT DISTINCT `{column_name}` FROM `{table_name}` WHERE `{column_name}` IS NOT NULL;")
                results = execute_sql(cursor,
                                      f"SELECT DISTINCT `{column_name}` FROM `{table_name}` WHERE `{column_name}` IS NOT NULL;")
                column_contents = [result[0] for result in results if
                                   isinstance(result[0], str) and not is_number(result[0])]

                for c_id, column_content in enumerate(column_contents):
                    if len(column_content) != 0 and len(column_content) <= 40:
                        all_column_contents.append(
                            {
                                "id": "{}-**-{}-**-{}".format(table_name, column_name, c_id),
                                "contents": column_content
                            }
                        )
            except Exception as e:
                print(str(e))

    temp_db_index_path = os.path.join(index_path, "temp_db_index")
    os.makedirs(temp_db_index_path, exist_ok=True)

    temp_content_path = f"{temp_db_index_path}/contents.json"

    with open(temp_content_path, "w") as f:
        f.write(json.dumps(all_column_contents, indent=2, ensure_ascii=True))

    os.makedirs(index_path, exist_ok=True)
    cmd = f'python -m pyserini.index.lucene --collection JsonCollection --input {temp_db_index_path} --index "{index_path}" --generator DefaultLuceneDocumentGenerator --threads 16 --storePositions --storeDocvectors --storeRaw'

    d = os.system(cmd)
    print(d)
    os.remove(temp_content_path)

def build_index_for_dataset(dataset_name: str, db_path: str, save_index_path: str):
    print(f"build index for dataset: {dataset_name} - {db_path}")
    print(f"save_index_path : {save_index_path}")
    remove_contents_of_a_folder(save_index_path)
    db_ids = os.listdir(db_path)
    for db_id in db_ids:
        db_file_path = os.path.join(db_path, db_id, db_id + ".sqlite")
        if os.path.exists(db_file_path) and os.path.isfile(db_file_path):
            print(f"db_id: {db_id}, The file '{db_file_path}' exists.")
            build_content_index(
                db_file_path,
                os.path.join(save_index_path, db_id)
            )
        else:
            print(f"The file '{db_file_path}' does not exist.")

def calculate_substring_match_percentage(query, target):
    query = query.lower()
    target = target.lower()

    substrings = []
    for i in range(len(query)):
        for j in range(i + 1, len(query) + 1):
            substrings.append(query[i:j])
    max_matched_substring_len = max([len(substring) for substring in substrings if substring in target])
    return max_matched_substring_len / len(query)

# --- 这是被修改的函数 ---
def retrieve_relevant_hits(searcher, queries):
    queries = list(dict.fromkeys(queries))
    q_ids = [f"{idx}" for idx in range(len(queries))]

    query2hits = dict()
    # searcher.batch_search 返回一个字典，key是q_id, value是ScoredDoc对象列表
    search_results = searcher.batch_search(queries, q_ids, k=10, threads=60)
    
    for query, q_id in zip(queries, q_ids):
        # search_results[q_id] 是一个包含 ScoredDoc 对象的列表
        scored_docs = search_results[q_id]
        
        # --- 修改部分开始 ---
        # 之前的错误代码尝试访问 hit.raw，但在新版pyserini中已不可用
        # hits = list(dict.fromkeys(([hit.raw for hit in hits])))
        
        # 正确的写法：
        # 1. 使用 searcher.doc(hit.docid).raw() 来获取每个文档的原始字符串内容
        # 2. 使用 dict.fromkeys 来高效去重
        raw_contents = list(dict.fromkeys([searcher.doc(hit.docid).raw() for hit in scored_docs]))
        # --- 修改部分结束 ---

        # 将去重后的JSON字符串列表解析成Python字典列表
        hits = [json.loads(raw_content) for raw_content in raw_contents]
        query2hits[query] = hits

    return query2hits
# --- 函数修改结束 ---

def retrieve_question_related_db_values(hits, question):
    high_score_hits = []
    for idx, hit in enumerate(hits):
        table_name, column_name, c_id = hit["id"].split("-**-")
        score = calculate_substring_match_percentage(hit["contents"], question)
        if score > 0.85:
            high_score_hits.append(
                {
                    "table_dot_column_lower_case": f"{table_name}.{column_name}".lower(),
                    "db_value": hit["contents"],
                    "score": score,
                    "index": idx,
                }
            )
    high_score_hits = sorted(high_score_hits, key=lambda x: (x["score"], len(x["db_value"]), x["index"]), reverse=True)
    high_score_hits = high_score_hits[:20]

    relavant_db_values_dict = dict()
    for hit in high_score_hits:
        if hit["table_dot_column_lower_case"] in relavant_db_values_dict:
            relavant_db_values_dict[hit["table_dot_column_lower_case"]].append(hit["db_value"])
        else:
            relavant_db_values_dict[hit["table_dot_column_lower_case"]] = [hit["db_value"]]

    return relavant_db_values_dict

def obtain_n_grams(sequence, max_n):
    tokens = word_tokenize(sequence)
    all_n_grams = []
    for n in range(1, max_n + 1):
        all_n_grams.extend([" ".join(gram) for gram in ngrams(tokens, n)])
    return all_n_grams

# 假设的主程序入口，用于演示代码完整性
if __name__ == '__main__':
    # 这里的代码是假设的，因为原始错误追溯中显示了对 process_dataset 的调用
    # 你需要根据你的实际项目结构来运行代码
    # 例如:
    # process_dataset(...)
    print("代码已修正。请在您的主程序中调用相应函数。")
    # 示例用法 (需要有已构建的索引和searcher对象)
    # try:
    #     searcher = LuceneSearcher('path/to/your/index')
    #     queries = ['some query text', 'another query']
    #     relevant_hits = retrieve_relevant_hits(searcher, queries)
    #     print(json.dumps(relevant_hits, indent=2))
    # except Exception as e:
    #     print(f"无法执行示例，请确保索引路径正确: {e}")

