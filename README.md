# DSR-SQL: Dual-Schema Reasoning for Text-to-SQL Framework

![dsrsql](image/DSR-SQL-Framework.png)

## Abstract

DSR-SQL (Dual-Schema Reasoning for Text-to-SQL) is an innovative dual-path collaborative reasoning framework designed to tackle the core conflict between "information completeness" and "reasoning accuracy" often faced by small open-source Large Language Models (LLMs) in complex Text-to-SQL tasks.

The framework generates candidate SQL queries by running two paths in parallel:

1.  **Full Schema Path (Path 1):** Directly utilizes the complete database schema. This ensures high recall and prevents the omission of critical information (the information masking effect).
2.  **Reduced Schema Path (Path 2):** Employs a dedicated Schema Selector to pre-filter relevant tables. This significantly reduces contextual noise and enhances reasoning precision.

Subsequently, DSR-SQL uses an intelligent fusion and selection mechanism (the SQLMerger/Selector), leveraging execution feedback, to analyze, merge, or select the optimal candidate SQL from the two paths. This approach successfully combines the benefits of high recall and high precision, leading to robust performance improvements for 7B-class open-source models on complex Text-to-SQL benchmarks.

---

## Code Execution Guide

**Important:** All commands below should be executed from the root directory of the DSR-SQL project repository.

```bash
# Navigate to the project root directory
cd /path/to/DSR-SQL
```

This guide details how to set up the environment and execute the DSR-SQL framework.

### 1. Data Preprocessing

The data preprocessing stage prepares the necessary inputs for model training and evaluation, including extracting golden table sets and generating the reduced schemas used by the framework.

#### 1.1 Create Preprocessing Environment

It is highly recommended to use Conda for environment management.

```bash
# Create a new Conda environment
conda create -n dsrsql_prep python=3.10
conda activate dsrsql_prep

# Install dependencies
# Assuming all preprocessing libraries are listed in requirements_prep.txt
pip install -r requirements_prep.txt
```

#### 1.2 Run Preprocessing Script

Execute the appropriate shell script based on the dataset you are using (Spider or BIRD). Before running, you **must** edit the script to point to your local dataset paths.

**General Note on Configuration:** The most critical parameters to modify are the dataset root paths. However, you should also review other path variables within the script to ensure they match your desired file structure.

**A. For Spider Dataset:**

 1. Open and edit the script `scripts/run_data_prep_spider.sh`.

 2. Modify the `SPIDER_DATASET_ROOT` variable to point to your local Spider data directory:

    ```bash
    # Example parameter to modify in the script:
    SPIDER_DATASET_ROOT="/path/to/your/Spider" 
    ```

 3. Run the script:

    ```bash
    sh scripts/run_data_prep_spider.sh
    ```

**B. For BIRD Dataset:**

1. Open and edit the script `scripts/run_data_prep_bird.sh` (assuming the BIRD script is named logically).

2. Modify the `BIRD_DATASET_ROOT` and `BIRD_MODE` variables:

   ```bash
   # Example parameters to modify in the script:
   BIRD_DATASET_ROOT="/path/to/your/BIRD"
   BIRD_MODE="train" # Set to 'train', 'dev', or 'test' as needed
   ```

3. Run the script:

   ```bash
   sh scripts/run_data_prep_bird.sh
   ```

### 2. Running the DSR-SQL Pipeline

This section covers the inference and evaluation phases, where the dual-path generation, refinement, and fusion logic of DSR-SQL is executed.

#### 2.1 Create Pipeline Environment

This environment requires configurations suitable for running Large Language Model (LLM) inference, including PyTorch and the Hugging Face Transformers library.

```bash
# Create a new Conda environment (if different from preprocessing)
conda create -n dsrsql python=3.10
conda activate dsrsql

# Step 1: Manually install PyTorch
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118

# Step 2: Manually install Flash Attention
pip install flash_attn/flash_attn-2.7.3+cu11torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# Step 3: Install the remaining dependencies from requirements.txt
pip install -r requirements.txt
```

#### 2.2 Run DSR-SQL Inference/Evaluation Script

The main logic of DSR-SQL (dual-path generation, SQL merging/selection, and execution) is executed via the pipeline script.

The final result after the execution of the DSR-SQL pipeline flow, specifically the optimal SQL query selected by the SQLMerger/Selector, will be saved to the following path: **`{OUTPUT_BASE_DIR}/{DATASET_NAME}/sql_results/selected_sqls.sql`。**

**Before running, you must modify the following parameters within `run_pipeline.sh`:**

| Parameter                 | Description                                                  | Example Value                                      |
| :------------------------ | :----------------------------------------------------------- | :------------------------------------------------- |
| `OUTPUT_BASE_DIR`         | The root directory where all pipeline outputs (logs, results, generated SQLs) will be saved. | `"outputs"`                                        |
| `DATASET_NAME`            | Used to create a subdirectory under `OUTPUT_BASE_DIR` for organizing results (e.g., `outputs-myqwen/bird/dev/`). | `"bird/dev"`                                       |
| `DB_SCHEMA_DIR`           | Path to the database schema description files generated during the preprocessing step (Step 1). | `"preprocess_data/bird/dev/db_schemas"`            |
| `DB_ROOT_DIR`             | Path to the actual database files (SQLite files) used for execution-based evaluation. | `"path/to/datasets/BIRD/dev/dev_databases"`        |
| `CSV_FILE_PATH`           | Path to the main preprocessed data file containing questions and metadata. | `"preprocess_data/bird/dev/processed_dataset.csv"` |
| `PIPELINE_CONFIGS_PATH`   | Path to the JSON configuration file defining the models (Table Selector, SQL Generator, Merger) and their parameters. | `"config/pipeline_configs.json"`                   |
| `MAX_SCHEMA_TOKEN_LENGTH` | Maximum token length allowed for the schema input. If the schema token length exceeds this value, the input is filtered or truncated. **If this parameter is omitted or not specified, schema length filtering will be disabled.** | `8192`                                             |

**Execution Command:**

```bash
# Execute the DSR-SQL pipeline
sh run_pipeline.sh
```

#### 2.3 Run Evaluation Scripts 

After the DSR-SQL pipeline completes, the evaluation scripts are used to calculate and compare the **Execution Accuracy (EX)** across the three core outputs generated by the framework:

1. **Full Schema Path (Path 1):** Predicted SQLs generated using the complete schema (`refined_sqls_1.sql`).
2. **Reduced Schema Path (Path 2):** Predicted SQLs generated using the schema selected by the Schema Selector (`refined_sqls_2.sql`).
3. **DSR-SQL Final Result:** The fused or selected optimal SQLs chosen by the SQLMerger/Selector (`selected_sqls.sql`).

This comparison allows for a direct assessment of the performance gain achieved by the DSR-SQL fusion mechanism over the individual paths.

##### 2.3.1 Spider Evaluation (`sh compare_spider_results.sh`)

This script is used to evaluate results on the Spider dataset.

| Parameter         | Description                                              | Example Value                         |
| :---------------- | :------------------------------------------------------- | :------------------------------------ |
| `OUTPUT_BASE_DIR` | The base directory containing the pipeline output files. | `"./outputs/spider"`                  |
| `data_mode`       | The subset of the data being evaluated.                  | `"dev"`                               |
| `DB_PATH`         | Path to the root directory of the Spider databases.      | `"/path/to/datasets/Spider/database"` |

**Execution Command:**

```bash
sh compare_spider_results.sh
```

##### 2.3.2 BIRD Evaluation (`sh compare_bird_results.sh`)

This script is used to evaluate results on the BIRD dataset, which requires additional difficulty information.

| Parameter         | Description                                              | Example Value                                |
| :---------------- | :------------------------------------------------------- | :------------------------------------------- |
| `OUTPUT_BASE_DIR` | The base directory containing the pipeline output files. | `"./outputs/bird"`                           |
| `data_mode`       | The subset of the data being evaluated.                  | `"dev"`                                      |
| `db_root_path`    | Path to the root directory of the BIRD databases.        | `"/path/to/datasets/BIRD/dev/dev_databases"` |
| `diff_json_path`  | Path to the BIRD JSON file containing difficulty labels. | `"/path/to/datasets/BIRD/dev/dev.json"`      |

**Execution Command:**

```bash
sh compare_bird_results.sh
```

