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