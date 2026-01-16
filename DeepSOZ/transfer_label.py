import pandas as pd
import numpy as np
import os


def convert_private_excel_to_tuh(source_excel_path, target_csv_template, output_csv_path):
    print(f"正在读取源文件: {source_excel_path} ...")
    # 1. 读取 Excel 文件
    # header=None 因为表头占据了前两行，我们需要手动处理
    df_source = pd.read_excel(source_excel_path, header=None)

    print(f"正在读取目标模板: {target_csv_template} ...")
    # 2. 读取目标 CSV 格式以获取列结构
    df_target_example = pd.read_csv(target_csv_template)
    # 获取目标文件的所有列名（去除 Unnamed 列）
    target_columns = [c for c in df_target_example.columns if 'Unnamed' not in c]

    # 3. 确定目标文件支持的通道
    # 您的标准19通道: ['fp1', 'fp2', 'f7', 'f3', 'fz', 'f4', 'f8', 't3', 'c3', 'cz','c4', 't4', 't5', 'p3', 'pz', 'p4', 't6', 'o1', 'o2']
    # TUH 格式通常还包含: ['oz', 'a1', 'a2']
    # 我们只填充目标文件中存在的列
    target_channel_cols = [
        'fp1', 'fp2', 'f7', 'f3', 'fz', 'f4', 'f8', 't3', 'c3', 'cz',
        'c4', 't4', 't5', 'p3', 'pz', 'p4', 't6', 'o1', 'o2',
        'oz', 'a1', 'a2'
    ]
    # 过滤出模板中实际存在的列
    valid_channel_cols = [c for c in target_channel_cols if c in target_columns]

    output_rows = []

    # 4. 遍历患者数据 (从第3行开始，即索引2，因为0和1是表头)
    for i in range(2, len(df_source)):
        row = df_source.iloc[i]

        name = row[0]  # 姓名
        if pd.isna(name): continue

        # 提取基础信息
        # 性别=row[1], 年龄=row[2], 致痫灶侧别=row[3]
        hemi_side = row[3]

        # 遍历4次可能的测试 (SZ1, SZ2, SZ3, SZ4)
        # 每次测试跨越4列。SZ1从第4列(索引4)开始
        # 结构: [起始, 显著电极, 早期扩散, 覆盖全导]
        for sz_idx in range(4):
            col_start = 4 + sz_idx * 4

            # 防止越界
            if col_start >= len(row): break

            # 获取关键字段
            start_desc = row[col_start]  # 起始
            electrodes_raw = row[col_start + 1]  # 显著电极
            spread_desc = row[col_start + 2]  # 早期扩散
            general_desc = row[col_start + 3]  # 覆盖全导

            # 如果显著电极和起始描述都为空，说明没有这次测试的数据
            if pd.isna(electrodes_raw) and pd.isna(start_desc):
                continue

            # --- 构建新的一行 ---
            new_entry = {col: np.nan for col in target_columns}

            # 填充元数据
            new_entry['pt_id'] = name
            new_entry['hemi'] = hemi_side
            new_entry['fn'] = f"{name}_SZ{sz_idx + 1}"  # 生成文件名标识
            new_entry['nsz'] = 1
            new_entry['nchns'] = 19  # 默认为19通道

            # 填充 Comments (包含所有原始描述，包括 SP1/SP2 信息)
            # 即使 SP1/SP2 不能映射到列，也会保留在这里
            comments_parts = [
                f"Original Electrodes: {electrodes_raw}",
                f"Start: {start_desc}",
                f"Spread: {spread_desc}",
                f"Generalized: {general_desc}"
            ]
            # 清理 None/NaN 值
            clean_comments = "; ".join([p for p in comments_parts if "nan" not in p])
            new_entry['Comments'] = clean_comments

            # 初始化所有通道列为 0
            for ch in valid_channel_cols:
                new_entry[ch] = 0

            # 解析显著电极并填充
            if isinstance(electrodes_raw, str):
                # 统一分隔符：处理中文逗号、顿号
                txt = electrodes_raw.replace('，', ',').replace('、', ',')
                # 分割并去除空格
                elec_list = [e.strip().lower() for e in txt.split(',')]

                for elec in elec_list:
                    # 再次清理可能的内部空格
                    elec_clean = elec.replace(' ', '')

                    # 尝试映射 SP1/SP2 到 Sph-L/Sph-R (如果目标表支持)
                    # 或者反之
                    if elec_clean == 'sph-l': elec_clean = 'sp1'
                    if elec_clean == 'sph-r': elec_clean = 'sp2'

                    # 检查该电极是否在目标格式的列中
                    if elec_clean in valid_channel_cols:
                        new_entry[elec_clean] = 1
                    else:
                        # 如果是 SP1/SP2 且目标表没有这两列，则忽略（已记录在 Comments 中）
                        pass

            output_rows.append(new_entry)

    # 5. 保存结果
    df_output = pd.DataFrame(output_rows, columns=target_columns)
    df_output.to_csv(output_csv_path, index=False,encoding='utf-8')
    print(f"转换完成！文件已保存至: {output_csv_path}")
    print(f"共处理了 {len(output_rows)} 条发作记录。")


# 使用示例 (请确保文件名与您本地文件一致)
if __name__ == '__main__':
    source_file = r'E:\DataSet\EEG\EEG dataset_SUAT\头皮扩散.xlsx'  # 您的Excel文件
    target_template = r'E:\code_learn\SUAT\workspace\EEG-projects\DeepSOZ\data\tuh_single_windowed_manifest.csv'  # 您的目标CSV文件
    output_file = 'converted_manifest.csv'  # 输出文件名

    if os.path.exists(source_file) and os.path.exists(target_template):
        convert_private_excel_to_tuh(source_file, target_template, output_file)
    else:
        print("请确保输入文件在当前目录下。")