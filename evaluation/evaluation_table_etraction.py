import pandas as pd

# 请将 'results.csv' 替换为实际的 CSV 文件名
df = pd.read_csv("./outputs/bak/7b/qwen/bird/dev/table_results/table_extraction_results.csv")

total_samples = len(df)
total_accuracy = 0
filtered_accuracy = 0
total_precision = 0
total_recall = 0

for index, row in df.iterrows():
    
    if pd.isna(row['predicted_tables']) or pd.isna(row['reference_tables']):
        print(f"Skipping row due to NaN values: Predicted='{row['predicted_tables']}', Reference='{row['reference_tables']}'")
        continue
    
    predicted_tables = str(row['predicted_tables']).split(", ")
    reference_tables = str(row['reference_tables']).split(", ")
    
    # Convert to lowercase and strip whitespace for comparison
    predicted_tables = [x.lower().replace("--","").replace("**","").strip() for x in predicted_tables]
    reference_tables = [x.lower().strip() for x in reference_tables]
    
    # Calculate accuracy
    if set(predicted_tables) == set(reference_tables):
        total_accuracy += 1
    
    # Calculate precision and recall
    true_positives = len(set(predicted_tables) & set(reference_tables))
    false_positives = len(set(predicted_tables) - set(reference_tables))
    false_negatives = len(set(reference_tables) - set(predicted_tables))

    if true_positives == len(reference_tables):
        filtered_accuracy += 1
    
    if len(predicted_tables) > 0:
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
    
    total_precision += precision
    total_recall += recall

# Calculate average precision and recall
avg_precision = total_precision / total_samples
avg_recall = total_recall / total_samples

# Calculate total accuracy
accuracy = total_accuracy / total_samples
filtered_accuracy = filtered_accuracy / total_samples

print("Total Accuracy:", accuracy)
print("Filtered Accuracy:", filtered_accuracy)
print("Average Precision:", avg_precision)
print("Average Recall:", avg_recall)
