import math

class Pharmacophore:
    def __init__(self, p_type, radius, x, y, z):
        self.p_type = p_type
        self.radius = float(radius)
        self.coords = (float(x), float(y), float(z))

def parse_phore_file(file_path):
    """
    解析 .phore 文件，自动跳过类型为 EX 的药效团
    """
    pharmacophores = []
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            # 跳过第一行标题（文件名）
            for line in lines[1:]:
                line = line.strip()
                if not line or line == "$$$$":
                    continue
                
                parts = line.split()
                if len(parts) >= 7:
                    p_type = parts[0]
                    
                    # --- 关键修改：过滤 EX 类型 ---
                    if p_type.upper() == "EX":
                        continue
                        
                    # 提取：第1列类型, 第3列半径, 第5-7列坐标
                    p = Pharmacophore(
                        p_type=p_type,
                        radius=parts[2],
                        x=parts[4],
                        y=parts[5],
                        z=parts[6]
                    )
                    pharmacophores.append(p)
    except Exception as e:
        print(f"读取文件 {file_path} 出错: {e}")
    return pharmacophores

def get_distance(c1, c2):
    """计算三维欧几里得距离"""
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(c1, c2)))

def calculate_overlap(ref_file, mol_file):
    """
    计算 mol 相对于 ref 的重合率
    """
    ref_list = parse_phore_file(ref_file)
    mol_list = parse_phore_file(mol_file)

    if not ref_list:
        print("ref 文件中没有药效团数据")
        return None, None, None
    
    if not mol_list:
        print("mol 文件中没有药效团数据")
        return  None, len(ref_list), None

    overlap_count = 0
    
    # 遍历 mol 中的每一个药效团
    for mol_p in mol_list:
        is_overlapped = False
        
        # 与 ref 中的每一个药效团比对
        for ref_p in ref_list:
            # 规则 1: 类型必须相同
            if mol_p.p_type == ref_p.p_type:
                # 规则 2: 距离小于等于 ref 的特征半径
                dist = get_distance(mol_p.coords, ref_p.coords)
                if dist <= ref_p.radius:
                    is_overlapped = True
                    break # 只要找到一个重合的 ref 点，该 mol 点就算重合
        
        if is_overlapped:
            overlap_count += 1
            
    overlap_rate = (overlap_count / len(ref_list)) * 100
    
    # print(f"--- 药效团重合率分析 ---")
    # print(f"Ref 总数: {len(ref_list)}")
    # print(f"Mol 总数: {len(mol_list)}")
    # print(f"重合个数: {overlap_count}")
    # print(f"重合率: {overlap_rate:.2f}%")
    
    return overlap_rate, len(ref_list), len(mol_list)

# 使用示例
if __name__ == "__main__":
    # 假设你的两个文件分别叫 ref.phore 和 mol.phore
    calculate_overlap("./data/charge_test/ABL2_HUMAN_274_551_0/ABL2_HUMAN_274_551_0.phore", 
                      "./data/charge_test/ABL2_HUMAN_274_551_0/ABL2_HUMAN_274_551_0_copy.phore")
    pass