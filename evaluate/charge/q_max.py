import pandas as pd
from pathlib import Path

def process_q_max_values(root_path, output_file):
    """
    遍历 root_path/*/*/q_value.csv，提取每个文件的 q 最大值，
    并按一级子文件夹去重。
    """
    root = Path(root_path)
    results = []

    for q_file in root.glob('*/*/q_local.csv'):
        try:
            level1_name = q_file.parents[1].name
            
            # 3. 读取 CSV 并获取 q 的最大值
            df_temp = pd.read_csv(q_file)
            df_temp = df_temp.iloc[:eval_num]
            
            if 'q' in df_temp.columns and not df_temp.empty:
                max_q = df_temp['q'].max()
                results.append({
                    "name": level1_name,
                    "q": max_q
                })
        except Exception as e:
            print(f"处理文件 {q_file} 时出错: {e}")

    final_df = pd.DataFrame(results)

    if final_df.empty:
        print("未找到有效数据。")
        return
    
    final_df = final_df.drop_duplicates(subset='name', keep='first')

    # 6. 保存为 CSV
    final_df.to_csv(output_file, index=False)
    print(f"数据已处理完成，结果保存至: {output_file}")
    return final_df

# --- 执行 ---
if __name__ == "__main__":
    ROOT_DIR = "./results/charge_local/diffsbdd/" 
    OUTPUT_CSV = f"{ROOT_DIR}/q_max.csv"

    eval_num = 5
    
    process_q_max_values(ROOT_DIR, OUTPUT_CSV)