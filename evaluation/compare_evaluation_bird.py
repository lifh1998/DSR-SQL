import os
import re
import sys
import json
import argparse
import sqlite3
import multiprocessing as mp
from func_timeout import func_timeout, FunctionTimedOut
import logging

# --- Copied functions from evaluation_bird_ex.py ---
def replace_multiple_spaces(text):
    # 定义正则表达式，匹配多个空字符
    pattern = r'\s+'
    # 将多个空字符替换成一个空格
    new_text = re.sub(pattern, ' ', text)
    return new_text

def load_json(dir):
    with open(dir, 'r', encoding='utf8') as j:
        contents = json.loads(j.read())
    return contents

def save_json_file(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"save json file to {path}")

# Global variable for result_callback. This will be managed within run_evaluation.
exec_result = [] 

def result_callback(result):
    exec_result.append(result)

def execute_sql(predicted_sql,ground_truth, db_path):
    conn = sqlite3.connect(db_path)
    # Connect to the database
    cursor = conn.cursor()
    cursor.execute(predicted_sql)
    predicted_res = cursor.fetchall()
    cursor.execute(ground_truth)
    ground_truth_res = cursor.fetchall()
    res = 0
    # todo: this should permute column order!
    if set(predicted_res) == set(ground_truth_res):
        res = 1
    return res

def execute_model(predicted_sql,ground_truth, db_place, idx, question_id, meta_time_out):
    try:
        res = func_timeout(meta_time_out, execute_sql,
                                  args=(predicted_sql, ground_truth, db_place))
    except KeyboardInterrupt:
        sys.exit(0)
    except FunctionTimedOut:
        result = [(f'timeout',)]
        res = 0
    except Exception as e:
        result = [(f'error',)]  # possibly len(query) > 512 or not executable
        res = 0
    result = {'sql_idx': idx, 'res': res, 'question_id': question_id}
    return result

def package_sqls(sql_path, db_root_path, mode='gpt', data_mode='dev'):
    clean_sqls = []
    db_path_list = []
    if mode == 'gpt':
        with open(sql_path, 'r', encoding='utf8') as sqls:
            sql_txt = sqls.readlines()
            for idx, sql_str in enumerate(sql_txt):
                clean_sqls.append(sql_str.strip())
                
    elif mode == 'gt':
        sqls = open(sql_path, encoding='utf8')
        sql_txt = sqls.readlines()
        for idx, sql_str in enumerate(sql_txt):
            sql, db_name = sql_str.strip().rsplit('\t', 1) # 以最后一个\t分割
            clean_sqls.append(sql)
            db_path_list.append(db_root_path + '/' + db_name + '/' + db_name + '.sqlite')
    return clean_sqls, db_path_list

def run_sqls_parallel(sqls_with_qids, db_places, num_cpus=1, meta_time_out=30.0):
    pool = mp.Pool(processes=num_cpus)
    for i, (sql_pair, question_id) in enumerate(sqls_with_qids):
        predicted_sql, ground_truth = sql_pair
        pool.apply_async(execute_model, args=(predicted_sql, ground_truth, db_places[i], i, question_id, meta_time_out), callback=result_callback)
    pool.close()
    pool.join()

def sort_results(list_of_dicts):
  return sorted(list_of_dicts, key=lambda x: x['sql_idx'])

def compute_acc_by_diff(exec_results, dev_contents):
    num_queries = len(exec_results)
    
    results = [res['res'] for res in exec_results]
    
    # Build a map from question_id to difficulty
    qid_to_difficulty = {str(content.get('question_id')): content.get('difficulty', 'simple') for content in dev_contents}

    simple_results, moderate_results, challenging_results = [], [], []

    for result in exec_results:
        question_id = result.get('question_id')
        if question_id not in qid_to_difficulty:
            logging.warning(f"Question ID {question_id} not found in dev.json. Skipping.")
            continue
        
        difficulty = qid_to_difficulty[question_id]
        
        if difficulty == 'simple':
            simple_results.append(result)
        elif difficulty == 'moderate':
            moderate_results.append(result)
        elif difficulty == 'challenging':
            challenging_results.append(result)

    simple_acc = sum([res['res'] for res in simple_results])/len(simple_results) if simple_results else 0
    moderate_acc = sum([res['res'] for res in moderate_results])/len(moderate_results) if moderate_results else 0
    challenging_acc = sum([res['res'] for res in challenging_results])/len(challenging_results) if challenging_results else 0
    
    all_acc = sum(results)/num_queries
    count_lists = [len(simple_results), len(moderate_results), len(challenging_results), num_queries]
    return simple_acc * 100, moderate_acc * 100, challenging_acc * 100, all_acc * 100, count_lists

def print_data(score_lists,count_lists):
    levels = ['simple', 'moderate', 'challenging', 'total']
    print("{:20} {:20} {:20} {:20} {:20}".format("", *levels))
    print("{:20} {:<20} {:<20} {:<20} {:<20}".format('count', *count_lists))

    print('======================================    ACCURACY    =====================================')
    print("{:20} {:<20.2f} {:<20.2f} {:<20.2f} {:<20.2f}".format('accuracy', *score_lists))

# --- New main logic for comparison ---
if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--predicted_sql_json_path_1', type=str, required=True, help='Path to the first predicted SQL file (refined_sqls_1.sql).')
    args_parser.add_argument('--predicted_sql_json_path_2', type=str, required=True, help='Path to the second predicted SQL file (refined_sqls_2.sql).')
    args_parser.add_argument('--predicted_sql_json_path_3', type=str, required=False, help='Path to the third predicted SQL file (selected_sqls.sql).')
    args_parser.add_argument('--ground_truth_sql_path', type=str, required=True, help='Path to the ground truth SQL file.')
    args_parser.add_argument('--data_mode', type=str, required=True, default='dev', choices=['train', 'dev', 'test'], help='Data mode (train, dev, or test).')
    args_parser.add_argument('--db_root_path', type=str, required=True, help='Root path to the databases.')
    args_parser.add_argument('--num_cpus', type=int, default=1, help='Number of CPUs to use for parallel execution.')
    args_parser.add_argument('--meta_time_out', type=float, default=30.0, help='Timeout for SQL execution.')
    args_parser.add_argument('--mode_predict', type=str, default='gpt', help='Prediction mode (e.g., gpt).')
    args_parser.add_argument('--difficulty',type=str, default='simple', help='Difficulty level (simple, moderate, challenging).')
    args_parser.add_argument('--diff_json_path',type=str,default='./data/bird/dev.json', help='Path to the difficulty JSON file.')
    args_parser.add_argument('--question_ids_path', type=str, required=True, help='Path to the question_ids.txt file.')
    args = args_parser.parse_args()

    # Function to run evaluation for a single model
    def run_evaluation(predicted_sql_path, ground_truth_sql_path, question_ids_path, db_root_path, data_mode, num_cpus, meta_time_out, mode_predict):
        global exec_result # Access the global variable
        exec_result = [] # Clear global variable for each run
        
        pred_queries, _ = package_sqls(predicted_sql_path, db_root_path, 
                                              mode=mode_predict, data_mode=data_mode)
        if len(pred_queries) == 0:
            raise ValueError(f'Empty data in {predicted_sql_path}')
        
        gt_queries, db_paths_gt = package_sqls(ground_truth_sql_path, db_root_path, mode='gt',
                                               data_mode=data_mode)

        # Load question_ids
        with open(question_ids_path, 'r', encoding='utf-8') as f:
            question_ids = [line.strip() for line in f.readlines()]

        assert len(pred_queries) == len(gt_queries), "len(pred_queries) != len(gt_queries)"
        assert len(pred_queries) == len(question_ids), "len(pred_queries) != len(question_ids)"

        # Combine sql_pairs with question_ids
        sqls_with_qids = list(zip(list(zip(pred_queries, gt_queries)), question_ids))
        
        run_sqls_parallel(sqls_with_qids, db_places=db_paths_gt, num_cpus=num_cpus, meta_time_out=meta_time_out)
        return sort_results(exec_result)

    # Load dev.json once
    dev_contents = load_json(args.diff_json_path)

    print("--- 评估 refined_sqls_1.sql ---")
    exec_result_1 = run_evaluation(args.predicted_sql_json_path_1, args.ground_truth_sql_path, 
                                   args.question_ids_path, 
                                   args.db_root_path, args.data_mode, args.num_cpus, 
                                   args.meta_time_out, args.mode_predict)
    
    print("\n--- 评估 refined_sqls_2.sql ---")
    exec_result_2 = run_evaluation(args.predicted_sql_json_path_2, args.ground_truth_sql_path, 
                                   args.question_ids_path, 
                                   args.db_root_path, args.data_mode, args.num_cpus, 
                                   args.meta_time_out, args.mode_predict)

    # 比较结果的辅助函数
    def compare_two_models(results1, results2, name1, name2):
        both_correct = 0
        both_incorrect = 0
        model1_correct_model2_incorrect = 0
        model1_incorrect_model2_correct = 0

        assert len(results1) == len(results2), f"评估结果长度不匹配: {name1} vs {name2}!"

        for i in range(len(results1)):
            res1 = results1[i]['res']
            res2 = results2[i]['res']

            if res1 == 1 and res2 == 1:
                both_correct += 1
            elif res1 == 0 and res2 == 0:
                both_incorrect += 1
            elif res1 == 1 and res2 == 0:
                model1_correct_model2_incorrect += 1
            elif res1 == 0 and res2 == 1:
                model1_incorrect_model2_correct += 1
        
        print(f"\n--- 比较结果: {name1} vs {name2} ---")
        print(f"两者都正确: {both_correct}")
        print(f"两者都错误: {both_incorrect}")
        print(f"{name1} 正确, {name2} 错误: {model1_correct_model2_incorrect}")
        print(f"{name1} 错误, {name2} 正确: {model1_incorrect_model2_correct}")
        print("--------------------------\n")

    if args.predicted_sql_json_path_3:
        print("\n--- 评估 selected_sqls.sql ---")
        exec_result_3 = run_evaluation(args.predicted_sql_json_path_3, args.ground_truth_sql_path, 
                                       args.question_ids_path, 
                                       args.db_root_path, args.data_mode, args.num_cpus, 
                                       args.meta_time_out, args.mode_predict)

        # 执行两两比较
        compare_two_models(exec_result_1, exec_result_2, "refined_sqls_1.sql", "refined_sqls_2.sql")
        compare_two_models(exec_result_1, exec_result_3, "refined_sqls_1.sql", "selected_sqls.sql")
        compare_two_models(exec_result_2, exec_result_3, "refined_sqls_2.sql", "selected_sqls.sql")

        # 打印每个模型的准确性
        print('\n--- refined_sqls_1.sql 准确性 ---')
        simple_acc_1, moderate_acc_1, challenging_acc_1, acc_1, count_lists_1 = \
            compute_acc_by_diff(exec_result_1, dev_contents)
        score_lists_1 = [simple_acc_1, moderate_acc_1, challenging_acc_1, acc_1]
        print_data(score_lists_1, count_lists_1)
        print('===========================================================================================')

        print('\n--- refined_sqls_2.sql 准确性 ---')
        simple_acc_2, moderate_acc_2, challenging_acc_2, acc_2, count_lists_2 = \
            compute_acc_by_diff(exec_result_2, dev_contents)
        score_lists_2 = [simple_acc_2, moderate_acc_2, challenging_acc_2, acc_2]
        print_data(score_lists_2, count_lists_2)
        print('===========================================================================================')

        print('\n--- selected_sqls.sql 准确性 ---')
        simple_acc_3, moderate_acc_3, challenging_acc_3, acc_3, count_lists_3 = \
            compute_acc_by_diff(exec_result_3, dev_contents)
        score_lists_3 = [simple_acc_3, moderate_acc_3, challenging_acc_3, acc_3]
        print_data(score_lists_3, count_lists_3)
        print('===========================================================================================')
    else:
        # 只执行两个模型的比较和准确性打印
        compare_two_models(exec_result_1, exec_result_2, "refined_sqls_1.sql", "refined_sqls_2.sql")

        print('\n--- refined_sqls_1.sql 准确性 ---')
        simple_acc_1, moderate_acc_1, challenging_acc_1, acc_1, count_lists_1 = \
            compute_acc_by_diff(exec_result_1, dev_contents)
        score_lists_1 = [simple_acc_1, moderate_acc_1, challenging_acc_1, acc_1]
        print_data(score_lists_1, count_lists_1)
        print('===========================================================================================')

        print('\n--- refined_sqls_2.sql 准确性 ---')
        simple_acc_2, moderate_acc_2, challenging_acc_2, acc_2, count_lists_2 = \
            compute_acc_by_diff(exec_result_2, dev_contents)
        score_lists_2 = [simple_acc_2, moderate_acc_2, challenging_acc_2, acc_2]
        print_data(score_lists_2, count_lists_2)
        print('===========================================================================================')
    
    print("完成比较评估")
