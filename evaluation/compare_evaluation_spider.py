################################
# val: number(float)/string(str)/sql(dict)
# col_unit: (agg_id, col_id, isDistinct(bool))
# val_unit: (unit_op, col_unit1, col_unit2)
# table_unit: (table_type, col_unit/sql)
# cond_unit: (not_op, op_id, val_unit, val1, val2)
# condition: [cond_unit1, 'and'/'or', cond_unit2, ...]
# sql {
#   'select': (isDistinct(bool), [(agg_id, val_unit), (agg_id, val_unit), ...])
#   'from': {'table_units': [table_unit1, table_unit2, ...], 'conds': condition}
#   'where': condition
#   'groupBy': [col_unit1, col_unit2, ...]
#   'orderBy': ('asc'/'desc', [val_unit1, val_unit2, ...])
#   'having': condition
#   'limit': None/limit value
#   'intersect': None/sql
#   'except': None/sql
#   'union': None/sql
# }
################################

import os
import json
import sqlite3
import argparse
from func_timeout import func_timeout, FunctionTimedOut # 导入超时机制

# Assuming process_sql and exec_eval are in the same directory or accessible via PYTHONPATH
from process_sql import get_schema, Schema, get_sql
from exec_eval import eval_exec_match

# Flag to disable value evaluation
DISABLE_VALUE = True
# Flag to disable distinct in select evaluation
DISABLE_DISTINCT = True


CLAUSE_KEYWORDS = ('select', 'from', 'where', 'group', 'order', 'limit', 'intersect', 'union', 'except')
JOIN_KEYWORDS = ('join', 'on', 'as')

WHERE_OPS = ('not', 'between', '=', '>', '<', '>=', '<=', '!=', 'in', 'like', 'is', 'exists')
UNIT_OPS = ('none', '-', '+', "*", '/')
AGG_OPS = ('none', 'max', 'min', 'count', 'sum', 'avg')
TABLE_TYPE = {
    'sql': "sql",
    'table_unit': "table_unit",
}

COND_OPS = ('and', 'or')
SQL_OPS = ('intersect', 'union', 'except')
ORDER_OPS = ('desc', 'asc')


HARDNESS = {
    "component1": ('where', 'group', 'order', 'limit', 'join', 'or', 'like'),
    "component2": ('except', 'union', 'intersect')
}


def condition_has_or(conds):
    return 'or' in conds[1::2]


def condition_has_like(conds):
    return WHERE_OPS.index('like') in [cond_unit[1] for cond_unit in conds[::2]]


def condition_has_sql(conds):
    for cond_unit in conds[::2]:
        val1, val2 = cond_unit[3], cond_unit[4]
        if val1 is not None and type(val1) is dict:
            return True
        if val2 is not None and type(val2) is dict:
            return True
    return False


def val_has_op(val_unit):
    return val_unit[0] != UNIT_OPS.index('none')


def has_agg(unit):
    return unit[0] != AGG_OPS.index('none')


def accuracy(count, total):
    if count == total:
        return 1
    return 0


def recall(count, total):
    if count == total:
        return 1
    return 0


def F1(acc, rec):
    if (acc + rec) == 0:
        return 0
    return (2. * acc * rec) / (acc + rec)


def get_scores(count, pred_total, label_total):
    if pred_total != label_total:
        return 0,0,0
    elif count == pred_total:
        return 1,1,1
    return 0,0,0


def eval_sel(pred, label):
    pred_sel = pred['select'][1]
    label_sel = label['select'][1]
    label_wo_agg = [unit[1] for unit in label_sel]
    pred_total = len(pred_sel)
    label_total = len(label_sel)
    cnt = 0
    cnt_wo_agg = 0

    for unit in pred_sel:
        if unit in label_sel:
            cnt += 1
            label_sel.remove(unit)
        if unit[1] in label_wo_agg:
            cnt_wo_agg += 1
            label_wo_agg.remove(unit[1])

    return label_total, pred_total, cnt, cnt_wo_agg


def eval_where(pred, label):
    pred_conds = [unit for unit in pred['where'][::2]]
    label_conds = [unit for unit in label['where'][::2]]
    label_wo_agg = [unit[2] for unit in label_conds]
    pred_total = len(pred_conds)
    label_total = len(label_conds)
    cnt = 0
    cnt_wo_agg = 0

    for unit in pred_conds:
        if unit in label_conds:
            cnt += 1
            label_conds.remove(unit)
        if unit[2] in label_wo_agg:
            cnt_wo_agg += 1
            label_wo_agg.remove(unit[2])

    return label_total, pred_total, cnt, cnt_wo_agg


def eval_group(pred, label):
    pred_cols = [unit[1] for unit in pred['groupBy']]
    label_cols = [unit[1] for unit in label['groupBy']]
    pred_total = len(pred_cols)
    label_total = len(label_cols)
    cnt = 0
    pred_cols = [pred.split(".")[1] if "." in pred else pred for pred in pred_cols]
    label_cols = [label.split(".")[1] if "." in label else label for label in label_cols]
    for col in pred_cols:
        if col in label_cols:
            cnt += 1
            label_cols.remove(col)
    return label_total, pred_total, cnt


def eval_having(pred, label):
    pred_total = label_total = cnt = 0
    if len(pred['groupBy']) > 0:
        pred_total = 1
    if len(label['groupBy']) > 0:
        label_total = 1

    pred_cols = [unit[1] for unit in pred['groupBy']]
    label_cols = [unit[1] for unit in label['groupBy']]
    if pred_total == label_total == 1 \
            and pred_cols == label_cols \
            and pred['having'] == label['having']:
        cnt = 1

    return label_total, pred_total, cnt


def eval_order(pred, label):
    pred_total = label_total = cnt = 0
    if len(pred['orderBy']) > 0:
        pred_total = 1
    if len(label['orderBy']) > 0:
        label_total = 1
    if len(label['orderBy']) > 0 and pred['orderBy'] == label['orderBy'] and \
            ((pred['limit'] is None and label['limit'] is None) or (pred['limit'] is not None and label['limit'] is not None)):
        cnt = 1
    return label_total, pred_total, cnt


def eval_and_or(pred, label):
    pred_ao = pred['where'][1::2]
    label_ao = label['where'][1::2]
    pred_ao = set(pred_ao)
    label_ao = set(label_ao)

    if pred_ao == label_ao:
        return 1,1,1
    return len(pred_ao),len(label_ao),0


def get_nestedSQL(sql):
    nested = []
    # Corrected slice to get actual condition units
    for cond_unit in sql['from']['conds'][::2] + sql['where'][::2] + sql['having'][::2]:
        # Ensure cond_unit has enough elements before accessing
        if len(cond_unit) > 3 and type(cond_unit[3]) is dict:
            nested.append(cond_unit[3])
        if len(cond_unit) > 4 and type(cond_unit[4]) is dict:
            nested.append(cond_unit[4])
    if sql['intersect'] is not None:
        nested.append(sql['intersect'])
    if sql['except'] is not None:
        nested.append(sql['except'])
    if sql['union'] is not None:
        nested.append(sql['union'])
    return nested


# Removed eval_nested and eval_IUEN as they are for EM


def get_keywords(sql):
    res = set()
    if len(sql['where']) > 0:
        res.add('where')
    if len(sql['groupBy']) > 0:
        res.add('group')
    if len(sql['having']) > 0:
        res.add('having')
    if len(sql['orderBy']) > 0:
        res.add(sql['orderBy'][0])
        res.add('order')
    if sql['limit'] is not None:
        res.add('limit')
    if sql['except'] is not None:
        res.add('except')
    if sql['union'] is not None:
        res.add('union')
    if sql['intersect'] is not None:
        res.add('intersect')

    # or keyword
    ao = sql['from']['conds'][1::2] + sql['where'][1::2] + sql['having'][1::2]
    if len([token for token in ao if token == 'or']) > 0:
        res.add('or')

    cond_units = sql['from']['conds'][::2] + sql['where'][::2] + sql['having'][::2]
    # not keyword
    if len([cond_unit for cond_unit in cond_units if cond_unit[0]]) > 0:
        res.add('not')

    # in keyword
    if len([cond_unit for cond_unit in cond_units if cond_unit[1] == WHERE_OPS.index('in')]) > 0:
        res.add('in')

    # like keyword
    if len([cond_unit for cond_unit in cond_units if cond_unit[1] == WHERE_OPS.index('like')]) > 0:
        res.add('like')

    return res


def eval_keywords(pred, label):
    pred_keywords = get_keywords(pred)
    label_keywords = get_keywords(label)
    pred_total = len(pred_keywords)
    label_total = len(label_keywords)
    cnt = 0

    for k in pred_keywords:
        if k in label_keywords:
            cnt += 1
    return label_total, pred_total, cnt


def count_agg(units):
    return len([unit for unit in units if has_agg(unit)])


def count_component1(sql):
    count = 0
    if len(sql['where']) > 0:
        count += 1
    if len(sql['groupBy']) > 0:
        count += 1
    if len(sql['orderBy']) > 0:
        count += 1
    if sql['limit'] is not None:
        count += 1
    if len(sql['from']['table_units']) > 0:  # JOIN
        count += len(sql['from']['table_units']) - 1

    ao = sql['from']['conds'][1::2] + sql['where'][1::2] + sql['having'][1::2]
    count += len([token for token in ao if token == 'or'])
    cond_units = sql['from']['conds'][::2] + sql['where'][::2] + sql['having'][::2]
    count += len([cond_unit for cond_unit in cond_units if cond_unit[1] == WHERE_OPS.index('like')])

    return count


def count_component2(sql):
    nested = get_nestedSQL(sql)
    return len(nested)


def count_others(sql):
    count = 0
    # number of aggregation
    agg_count = count_agg(sql['select'][1])
    agg_count += count_agg(sql['where'][::2])
    agg_count += count_agg(sql['groupBy'])
    if len(sql['orderBy']) > 0:
        agg_count += count_agg([unit[1] for unit in sql['orderBy'][1] if unit[1]] +
                            [unit[2] for unit in sql['orderBy'][1] if unit[2]])
    agg_count += count_agg(sql['having'])
    if agg_count > 1:
        count += 1

    # number of select columns
    if len(sql['select'][1]) > 1:
        count += 1

    # number of where conditions
    if len(sql['where']) > 1:
        count += 1

    # number of group by clauses
    if len(sql['groupBy']) > 1:
        count += 1

    return count


class Evaluator:
    """A simple evaluator"""
    def __init__(self):
        self.partial_scores = None

    def eval_hardness(self, sql):
        count_comp1_ = count_component1(sql)
        count_comp2_ = count_component2(sql)
        count_others_ = count_others(sql)

        if count_comp1_ <= 1 and count_others_ == 0 and count_comp2_ == 0:
            return "easy"
        elif (count_others_ <= 2 and count_comp1_ <= 1 and count_comp2_ == 0) or \
                (count_comp1_ <= 2 and count_others_ < 2 and count_comp2_ == 0):
            return "medium"
        elif (count_others_ > 2 and count_comp1_ <= 2 and count_comp2_ == 0) or \
                (2 < count_comp1_ <= 3 and count_others_ <= 2 and count_comp2_ == 0) or \
                (count_comp1_ <= 1 and count_others_ == 0 and count_comp2_ <= 1):
            return "hard"
        else:
            return "extra"

    # Removed eval_exact_match and eval_partial_match as per user request to ignore EM.


def isValidSQL(sql, db):
    conn = sqlite3.connect(db)
    cursor = conn.cursor()
    try:
        cursor.execute(sql)
    except:
        return False
    return True


def print_formated_s(row_name, l, element_format):
    template = "{:20} " + ' '.join([element_format] * len(l))
    print(template.format(row_name, *l))


def print_scores(scores, etype, include_turn_acc=True):
    turns = ['turn 1', 'turn 2', 'turn 3', 'turn 4', 'turn > 4']
    levels = ['easy', 'medium', 'hard', 'extra', 'all'] # Removed 'joint_all'
    if include_turn_acc:
        levels.append('joint_all') # This will add 'joint_all' twice if already in levels, but it's fine for now.

    print_formated_s("", levels, '{:20}')
    counts = [scores[level]['count'] for level in levels]
    print_formated_s("count", counts, '{:<20d}')

    if etype in ["exec"]: # Only print exec accuracy
        print ('=====================   EXECUTION ACCURACY     =====================')
        exec_scores = [scores[level]['exec'] for level in levels]
        print_formated_s("execution", exec_scores, '{:<20.3f}')

    if include_turn_acc:
        print()
        print()
        print_formated_s("", turns, '{:20}')
        counts = [scores[turn]['count'] for turn in turns]
        print_formated_s("count", counts, "{:<20d}")

        if etype in ["exec"]: # Only print turn exec accuracy
            print ('=====================   TURN EXECUTION ACCURACY     =====================')
            exec_scores = [scores[turn]['exec'] for turn in turns]
            print_formated_s("execution", exec_scores, '{:<20.3f}')


def execute_sql_with_timeout(db_path, p_str, g_str, plug_value, keep_distinct, progress_bar_for_each_datapoint, meta_time_out):
    try:
        res = func_timeout(meta_time_out, eval_exec_match,
                           kwargs={
                               'db': db_path,
                               'p_str': p_str,
                               'g_str': g_str,
                               'plug_value': plug_value,
                               'keep_distinct': keep_distinct,
                               'progress_bar_for_each_datapoint': progress_bar_for_each_datapoint
                           })
    except FunctionTimedOut:
        res = False # Timeout, consider it incorrect
    except Exception as e:
        res = False # Error during execution, consider it incorrect
    return res

def evaluate_single_model(gold, predict, db_dir, etype, plug_value, keep_distinct, progress_bar_for_each_datapoint, meta_time_out, question_ids):
    with open(gold) as f:
        glist = []
        gseq_one = []
        for l in f.readlines():
            if len(l.strip()) == 0:
                glist.append(gseq_one)
                gseq_one = []
            else:
                lstrip = l.strip().split('\t')
                gseq_one.append(lstrip)

        if len(gseq_one) != 0:
            glist.append(gseq_one)

    include_turn_acc = len(glist) > 1

    with open(predict) as f:
        plist = []
        pseq_one = []
        for l in f.readlines():
            if len(l.strip()) == 0:
                plist.append(pseq_one)
                pseq_one = []
            else:
                pseq_one.append(l.strip().split('\t'))

        if len(pseq_one) != 0:
            plist.append(pseq_one)

    assert len(plist) == len(glist), "number of sessions must equal"
    assert len(question_ids) == sum(len(seq) for seq in glist), "number of question_ids must match total queries"

    evaluator = Evaluator()
    turns = ['turn 1', 'turn 2', 'turn 3', 'turn 4', 'turn > 4']
    levels = ['easy', 'medium', 'hard', 'extra', 'all']

    entries = []
    scores = {}

    for turn in turns:
        scores[turn] = {'count': 0, 'exec': 0.}
    
    for level in levels:
        scores[level] = {'count': 0, 'exec': 0.}

    gold_pred_map_lst = []
    
    qid_idx = 0 # Index for question_ids list
    for i, (p, g) in enumerate(zip(plist, glist)):
        if (i + 1) % 10 == 0:
            print('Evaluating %dth prediction' % (i + 1))
        turn_scores = {"exec": []}
        
        for idx, pg in enumerate(zip(p, g)):
            gold_pred_map = {
                'idx': idx,
                'db_id': '',
                'question': '',
                'gold': '',
                'pred': '',
                'exec_result': 0,
                'question_id': '' # Added question_id
            }
            p_str_list, g_str_list = pg
            p_str = p_str_list[0]
            p_str = p_str.replace("value", "1")
            g_str, db, *_ = g_str_list

            gold_pred_map['pred'] = p_str
            gold_pred_map['gold'] = g_str
            gold_pred_map['db_id'] = db
            gold_pred_map['question_id'] = question_ids[qid_idx] # Assign question_id
            qid_idx += 1

            db_name = db
            db_path = os.path.join(db_dir, db, db + ".sqlite")
            schema = Schema(get_schema(db_path))
            g_sql = get_sql(schema, g_str)
            hardness = evaluator.eval_hardness(g_sql)
            if idx > 3:
                idx_turn = "> 4"
            else:
                idx_turn = str(idx + 1)
            turn_id = "turn " + str(idx_turn)
            scores[turn_id]['count'] += 1
            scores[hardness]['count'] += 1
            scores['all']['count'] += 1

            try:
                p_sql = get_sql(schema, p_str)
            except:
                p_sql = {
                "except": None, "from": {"conds": [], "table_units": []}, "groupBy": [], "having": [],
                "intersect": None, "limit": None, "orderBy": [], "select": [False, []], "union": None, "where": []
                }

            if etype in ["exec"]:
                exec_score = execute_sql_with_timeout(db_path=db_path, p_str=p_str, g_str=g_str, plug_value=plug_value,
                                             keep_distinct=keep_distinct, progress_bar_for_each_datapoint=progress_bar_for_each_datapoint,
                                             meta_time_out=meta_time_out)
                if exec_score:
                    scores[hardness]['exec'] += 1
                    scores[turn_id]['exec'] += 1
                    scores['all']['exec'] += 1
                    turn_scores['exec'].append(1)
                    gold_pred_map['exec_result'] = 1
                else:
                    turn_scores['exec'].append(0)
                gold_pred_map_lst.append(gold_pred_map)
            
        if all(v == 1 for v in turn_scores["exec"]):
            pass

    for turn in turns:
        if scores[turn]['count'] == 0:
            continue
        if etype in ["exec"]:
            scores[turn]['exec'] /= scores[turn]['count']

    for level in levels:
        if scores[level]['count'] == 0:
            continue
        if etype in ["exec"]:
            scores[level]['exec'] /= scores[level]['count']

    return gold_pred_map_lst, scores, include_turn_acc


# Removed all rebuild_sql_val and rebuild_sql_col related functions
# Removed build_foreign_key_map and build_foreign_key_map_from_json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gold', dest='gold', type=str, help="the path to the gold queries")
    parser.add_argument('--pred1', dest='pred1', type=str, help="the path to the first predicted queries file")
    parser.add_argument('--pred2', dest='pred2', type=str, help="the path to the second predicted queries file")
    parser.add_argument('--pred3', dest='pred3', type=str, required=False, help="the path to the third predicted queries file") # Added pred3
    parser.add_argument('--db', dest='db', type=str, help="the directory that contains all the databases and test suites")
    parser.add_argument('--table', dest='table', type=str, help="the tables.json schema file (not used for exec evaluation)")
    parser.add_argument('--etype', dest='etype', type=str, default='exec',
                        help="evaluation type, exec for test suite accuracy",
                        choices=('exec',))
    parser.add_argument('--plug_value', default=False, action='store_true',
                        help='whether to plug in the gold value into the predicted query; suitable if your model does not predict values.')
    parser.add_argument('--keep_distinct', default=False, action='store_true',
                        help='whether to keep distinct keyword during evaluation. default is false.')
    parser.add_argument('--progress_bar_for_each_datapoint', default=False, action='store_true',
                        help='whether to print progress bar of running test inputs for each datapoint')
    parser.add_argument('--question_ids_path', dest='question_ids_path', type=str, required=True, help='Path to the question_ids.txt file.') # Added question_ids_path
    parser.add_argument('--meta_time_out', type=float, default=30.0, help='Timeout for SQL execution.') # Added meta_time_out
    args = parser.parse_args()

    # Load question_ids
    with open(args.question_ids_path, 'r', encoding='utf-8') as f:
        question_ids = [line.strip() for line in f.readlines()]

    # Helper function for comparison
    def compare_two_models(results1, results2, name1, name2):
        both_correct = 0
        both_incorrect = 0
        model1_correct_model2_incorrect = 0
        model1_incorrect_model2_correct = 0

        assert len(results1) == len(results2), f"Evaluation results length mismatch: {name1} vs {name2}!"

        for i in range(len(results1)):
            res1 = results1[i]['exec_result']
            res2 = results2[i]['exec_result']

            if res1 == 1 and res2 == 1:
                both_correct += 1
            elif res1 == 0 and res2 == 0:
                both_incorrect += 1
            elif res1 == 1 and res2 == 0:
                model1_correct_model2_incorrect += 1
            elif res1 == 0 and res2 == 1:
                model1_incorrect_model2_correct += 1
        
        print(f"\n--- Comparison Results: {name1} vs {name2} ---")
        print(f"Both correct: {both_correct}")
        print(f"Both incorrect: {both_incorrect}")
        print(f"{name1} correct, {name2} incorrect: {model1_correct_model2_incorrect}")
        print(f"{name1} incorrect, {name2} correct: {model1_incorrect_model2_correct}")
        print("--------------------------\n")

    print("--- Evaluating Model 1 ---")
    gold_pred_map_lst_1, scores_1, include_turn_acc_1 = evaluate_single_model(
        args.gold, args.pred1, args.db, args.etype, args.plug_value, args.keep_distinct, args.progress_bar_for_each_datapoint, args.meta_time_out, question_ids
    )
    print_scores(scores_1, args.etype, include_turn_acc=include_turn_acc_1)
    print('===========================================================================================')

    print("\n--- Evaluating Model 2 ---")
    gold_pred_map_lst_2, scores_2, include_turn_acc_2 = evaluate_single_model(
        args.gold, args.pred2, args.db, args.etype, args.plug_value, args.keep_distinct, args.progress_bar_for_each_datapoint, args.meta_time_out, question_ids
    )
    print_scores(scores_2, args.etype, include_turn_acc=include_turn_acc_2)
    print('===========================================================================================')

    if args.pred3:
        print("\n--- Evaluating Model 3 ---")
        gold_pred_map_lst_3, scores_3, include_turn_acc_3 = evaluate_single_model(
            args.gold, args.pred3, args.db, args.etype, args.plug_value, args.keep_distinct, args.progress_bar_for_each_datapoint, args.meta_time_out, question_ids
        )
        print_scores(scores_3, args.etype, include_turn_acc=include_turn_acc_3)
        print('===========================================================================================')

        compare_two_models(gold_pred_map_lst_1, gold_pred_map_lst_2, "Model 1", "Model 2")
        compare_two_models(gold_pred_map_lst_1, gold_pred_map_lst_3, "Model 1", "Model 3")
        compare_two_models(gold_pred_map_lst_2, gold_pred_map_lst_3, "Model 2", "Model 3")
    else:
        compare_two_models(gold_pred_map_lst_1, gold_pred_map_lst_2, "Model 1", "Model 2")
