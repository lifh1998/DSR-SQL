table_extraction_prompt = """### Task Description
Given the following database schema, your job is to determine the tables that may be involved in answering the question.

### Database Schema
{database_schema}

### Question
{question}

### Response Format
Output the response in the following format:
<answer>
<table> table_1 </table>
<table> table_2 </table>
...
<table> table_n </table>
</answer>
"""

table_extraction_response = """<answer>
<table> {correct_tables} </table>
</answer>
"""

sql_generation_prompt = """### Task Description
Given the following database schema, your job is to generate the Sqlite SQL query given the user's question.

### Database Schema
{database_schema}

### Question
{question}

### Response Format
Output the response in the following format:
```
<answer>
<sql> SELECT ... </sql>
</answer>
```
"""

sql_generation_response = """<answer>
<sql> {query} </sql>
</answer>
"""

sql_refinement_prompt = """### Task Description
Given the database schema below, the original question, a candidate SQL query, and an error message from database execution, refine the candidate SQL query to fix the error and make it executable.

### Database Schema
{database_schema}

### Question
{question}

### Candidate SQL
{candidate_sql}

### Error Message
{error_message}

### Response Format
Output the response in the following format:
```
<answer>
<sql> SELECT ... </sql>
</answer>
```
"""

sql_selection_classifier_prompt = """### Task Description
Given the original question, the database schema, and two candidate SQL queries, select the best SQL query that answers the question.

### Database Schema
{database_schema}

### Question
{question}

### Candidate SQL
SQL 1: {candidate_sql_1}
SQL 2: {candidate_sql_2}
"""

sql_selection_prompt = f"""{sql_selection_classifier_prompt}

### Response Format
Output the response in the following format:
```
<answer>
<sql> SELECT ... </sql>
</answer>
```
"""

cscsql_generation_prompt = """You first thinks about the reasoning process in the mind and then provides the user with the answer.

Task Overview:
You are a data science expert. Below, you are provided with a database schema and a natural language question. Your task is to understand the schema and generate a valid SQL query to answer the question.

Database Engine:
SQLite

Database Schema:
{database_schema}
This schema describes the database's structure, including tables, columns, primary keys, foreign keys, and any relevant relationships or constraints.

Question:
{question}

Instructions:
- Make sure you only output the information that is asked in the question. If the question asks for a specific column, make sure to only include that column in the SELECT clause, nothing more.
- The generated query should return all of the information asked in the question without any missing or extra information.
- Before generating the final SQL query, please think through the steps of how to write the query.

Output Format:
Show your work in <think> </think> tags. And return the final SQLite SQL query that starts with keyword `SELECT` in <answer> </answer> tags, \
for example <answer>SELECT AVG(rating_score) FROM movies</answer>.

Let me solve this step by step.
"""

cscsql_merge_prompt = """You first thinks about the reasoning process in the mind and then provides the user with the answer.

Task Overview:
You are a data science expert. Below, you are provided with a database schema, a natural language question, some draft SQL and its corresponding execution result. Your task is to understand the schema and generate a valid SQL query to answer the question.

Database Engine:
SQLite

Database Schema:
{database_schema}
This schema describes the database's structure, including tables, columns, primary keys, foreign keys, and any relevant relationships or constraints.

Question:
{question}

Here are some corresponding draft SQL and execute result:
1. {candidate_sql_1}
【Execution result】
{sql1_exec_results}

2. {candidate_sql_2}
【Execution result】
{sql2_exec_results}

Instructions:
- Make sure you only output the information that is asked in the question. If the question asks for a specific column, make sure to only include that column in the SELECT clause, nothing more.
- The generated query should return all of the information asked in the question without any missing or extra information.
- Before generating the final SQL query, please think through the steps of how to write the query.

Output Format:
Show your work in <think> </think> tags. And return the final SQLite SQL query that starts with keyword `SELECT` in <answer> </answer> tags, \
for example <answer>SELECT AVG(rating_score) FROM movies</answer>.

Let me solve this step by step.
"""

cscsql_system_prompt = "You are a helpful AI Assistant that provides well-reasoned and detailed responses. You first think about the reasoning process as an internal monologue and then provide the user with the answer. Respond in the following format: <think>\n...\n</think>\n<answer>\n...\n</answer>"
