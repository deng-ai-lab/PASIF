import pandas as pd
from pathlib import Path

def process_q_max_values(root_path, output_file):
    """
    遍历 root_path/*/*/q_value.csv，提取每个文件的 q 最大值，
    并按一级子文件夹去重。
    """
    root = Path(root_path)
    results = []

    # 1. 匹配二级子文件夹下的 q_value.csv
    # 路径结构为：root / 一级文件夹 / 二级文件夹 / q_value.csv
    for q_file in root.glob('*/*/q_local.csv'):
        try:
            # 2. 提取一级子文件夹名称
            # q_file.parents[0] 是二级文件夹，parents[1] 是一级文件夹
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

    # 4. 转化为 DataFrame
    final_df = pd.DataFrame(results)

    if final_df.empty:
        print("未找到有效数据。")
        return

    # 5. 处理重复的一级子文件夹行
    # 若一级文件夹重复，drop_duplicates 默认保留第一行。
    # 如果你想保留所有记录中 q 最大的那一行，可以先 sort_values 再去重。
    # final_df = final_df.sort_values(by='Max_Q', ascending=False) # 按 q 降序排
    final_df = final_df.drop_duplicates(subset='name', keep='first')

    # 6. 保存为 CSV
    final_df.to_csv(output_file, index=False)
    print(f"数据已处理完成，结果保存至: {output_file}")
    return final_df

# --- 执行 ---
if __name__ == "__main__":
    # 设定你的根目录和输出文件名
    ROOT_DIR = "./results/charge_local/diffsbdd/" 
    OUTPUT_CSV = f"{ROOT_DIR}/q_max.csv"

    eval_num = 5
    
    process_q_max_values(ROOT_DIR, OUTPUT_CSV)