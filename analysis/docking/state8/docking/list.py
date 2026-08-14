#!/usr/bin/env python3
"""
AutoDock Vina 结果统计与排序脚本
提取out文件夹下所有结果的对接分数，排序后生成完整记录文件
"""

import os
import re
import csv
import time
from pathlib import Path
from datetime import datetime

def extract_vina_score_from_file(file_path: Path) -> float:
    """
    从PDBQT文件中提取Vina对接分数
    
    参数:
        file_path: PDBQT文件路径
        
    返回:
        对接分数（浮点数），如果未找到则返回None
    """
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if "VINA RESULT:" in line:
                    # 使用正则表达式提取第一个浮点数（对接分数）
                    match = re.search(r'[-+]?\d*\.\d+', line)
                    if match:
                        return float(match.group())
        return None
    except Exception as e:
        print(f"  读取文件错误: {e}")
        return None

def extract_additional_info_from_file(file_path: Path) -> dict:
    """
    从PDBQT文件中提取额外信息
    
    参数:
        file_path: PDBQT文件路径
        
    返回:
        包含额外信息的字典
    """
    info = {
        'inter_intra': None,
        'inter': None,
        'intra': None,
        'unbound': None,
        'ligand_name': None,
        'active_torsions': None
    }
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read(2000)  # 只读取前2000个字符
            
            # 提取INTER + INTRA
            match = re.search(r'INTER \+ INTRA:\s*([-+]?\d*\.\d+)', content)
            if match:
                info['inter_intra'] = float(match.group(1))
            
            # 提取INTER
            match = re.search(r'INTER:\s*([-+]?\d*\.\d+)', content)
            if match:
                info['inter'] = float(match.group(1))
            
            # 提取INTRA
            match = re.search(r'INTRA:\s*([-+]?\d*\.\d+)', content)
            if match:
                info['intra'] = float(match.group(1))
            
            # 提取UNBOUND
            match = re.search(r'UNBOUND:\s*([-+]?\d*\.\d+)', content)
            if match:
                info['unbound'] = float(match.group(1))
            
            # 提取配体名称
            match = re.search(r'Name\s*=\s*(\S+)', content)
            if match:
                info['ligand_name'] = match.group(1)
            
            # 提取活性扭转数
            match = re.search(r'(\d+)\s+active torsions:', content)
            if match:
                info['active_torsions'] = int(match.group(1))
    
    except Exception as e:
        print(f"  提取额外信息错误: {e}")
    
    return info

def generate_complete_report(results, output_file="vina_complete_report.txt"):
    """
    生成完整的报告文件，包含所有结果和统计信息
    
    参数:
        results: 排序后的结果列表
        output_file: 输出文件名
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        # 写入文件头
        f.write("=" * 120 + "\n")
        f.write("AUTODOCK VINA 对接结果完整报告\n")
        f.write("=" * 120 + "\n")
        f.write(f"生成时间: {timestamp}\n")
        f.write(f"结果总数: {len(results)}\n")
        f.write("=" * 120 + "\n\n")
        
        # 写入统计摘要
        f.write("统计摘要:\n")
        f.write("-" * 120 + "\n")
        
        if results:
            scores = [r['score'] for r in results]
            min_score = min(scores)
            max_score = max(scores)
            avg_score = sum(scores) / len(scores)
            
            f.write(f"最佳对接分数: {min_score:.3f} kcal/mol\n")
            f.write(f"最差对接分数: {max_score:.3f} kcal/mol\n")
            f.write(f"平均对接分数: {avg_score:.3f} kcal/mol\n")
            f.write(f"中位数分数: {sorted(scores)[len(scores)//2]:.3f} kcal/mol\n")
            
            # 分数分布
            f.write("\n分数分布:\n")
            ranges = [
                (-float('inf'), -10, "< -10"),
                (-10, -8, "-10 ~ -8"),
                (-8, -6, "-8 ~ -6"),
                (-6, -4, "-6 ~ -4"),
                (-4, 0, "-4 ~ 0"),
                (0, float('inf'), "> 0")
            ]
            
            for low, high, label in ranges:
                count = sum(1 for s in scores if low <= s < high)
                percentage = count / len(scores) * 100
                f.write(f"  {label:<12}: {count:>3} 个 ({percentage:>5.1f}%)\n")
        
        f.write("\n" + "=" * 120 + "\n")
        f.write("完整结果列表 (按对接分数排序，从最佳到最差):\n")
        f.write("=" * 120 + "\n\n")
        
        # 写入表头
        header = f"{'排名':<6} {'文件夹名':<20} {'对接分数':<12} {'INTER':<10} {'INTRA':<10} {'配体名称':<25} {'活性扭转':<8} {'文件名':<20}"
        f.write(header + "\n")
        f.write("-" * 120 + "\n")
        
        # 写入所有结果
        for i, result in enumerate(results, 1):
            folder = result['folder']
            if len(folder) > 19:
                folder = folder[:17] + "..."
            
            ligand_name = result.get('ligand_name', 'N/A')
            if len(ligand_name) > 24:
                ligand_name = ligand_name[:22] + "..."
            
            filename = result['file']
            if len(filename) > 19:
                filename = filename[:17] + "..."
            
            inter = result.get('inter', 'N/A')
            if inter != 'N/A':
                inter = f"{inter:.3f}"
            
            intra = result.get('intra', 'N/A')
            if intra != 'N/A':
                intra = f"{intra:.3f}"
            
            active_torsions = result.get('active_torsions', 'N/A')
            
            line = f"{i:<6} {folder:<20} {result['score']:<12.3f} {inter:<10} {intra:<10} {ligand_name:<25} {active_torsions:<8} {filename:<20}"
            f.write(line + "\n")
        
        f.write("\n" + "=" * 120 + "\n")
        f.write("文件结束\n")
    
    print(f"完整报告已保存到: {output_file}")

def main():
    print("=" * 80)
    print("AutoDock Vina 对接结果统计与排序工具")
    print("=" * 80)
    print("正在扫描out文件夹...")
    
    results = []
    out_dir = Path("out")
    
    if not out_dir.exists():
        print("错误: out文件夹不存在")
        print("请确保脚本与out文件夹在同一目录下")
        return
    
    # 获取所有子文件夹
    folders = sorted([f for f in out_dir.iterdir() if f.is_dir()])
    
    if not folders:
        print("错误: out文件夹中没有子文件夹")
        return
    
    print(f"找到 {len(folders)} 个子文件夹")
    
    start_time = time.time()
    
    # 遍历所有子文件夹
    for i, folder in enumerate(folders, 1):
        print(f"[{i:3d}/{len(folders)}] 处理: {folder.name}")
        
        # 查找PDBQT文件
        pdbqt_files = list(folder.glob("*.pdbqt"))
        
        if not pdbqt_files:
            print(f"  警告: 未找到.pdbqt文件")
            continue
        
        # 通常每个文件夹只有一个PDBQT文件，取第一个
        pdbqt_file = pdbqt_files[0]
        
        # 提取对接分数
        score = extract_vina_score_from_file(pdbqt_file)
        
        if score is None:
            print(f"  警告: 未找到对接分数")
            continue
        
        # 提取额外信息
        extra_info = extract_additional_info_from_file(pdbqt_file)
        
        # 创建结果字典
        result = {
            'folder': folder.name,
            'score': score,
            'file': pdbqt_file.name,
            'full_path': str(pdbqt_file)
        }
        
        # 添加额外信息
        result.update(extra_info)
        
        results.append(result)
        print(f"  成功: 分数 = {score:.3f}")
    
    if not results:
        print("未找到任何有效的对接结果")
        return
    
    elapsed_time = time.time() - start_time
    print(f"\n处理完成! 耗时: {elapsed_time:.2f}秒")
    print(f"成功提取 {len(results)} 个对接结果")
    
    # 按分数排序（升序，因为负值越小越好）
    print("正在按对接分数排序...")
    results.sort(key=lambda x: x['score'])
    
    # 1. 生成CSV文件（便于导入Excel）
    print("生成CSV文件...")
    csv_filename = "vina_results_sorted.csv"
    with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
        # 定义字段
        fieldnames = ['rank', 'folder_name', 'score', 'inter', 'intra', 
                     'inter_intra', 'unbound', 'ligand_name', 
                     'active_torsions', 'file_name', 'full_path']
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for i, result in enumerate(results, 1):
            row = {
                'rank': i,
                'folder_name': result['folder'],
                'score': f"{result['score']:.3f}",
                'inter': result.get('inter', 'N/A'),
                'intra': result.get('intra', 'N/A'),
                'inter_intra': result.get('inter_intra', 'N/A'),
                'unbound': result.get('unbound', 'N/A'),
                'ligand_name': result.get('ligand_name', 'N/A'),
                'active_torsions': result.get('active_torsions', 'N/A'),
                'file_name': result['file'],
                'full_path': result['full_path']
            }
            writer.writerow(row)
    
    print(f"CSV文件已保存到: {csv_filename}")
    
    # 2. 生成完整报告文件
    print("生成完整报告文件...")
    txt_filename = "vina_results_complete_report.txt"
    generate_complete_report(results, txt_filename)
    
    # 3. 生成简明的排名文件
    print("生成简明排名文件...")
    simple_filename = "vina_results_ranking.txt"
    with open(simple_filename, 'w', encoding='utf-8') as f:
        f.write("排名\t文件夹名\t对接分数(kcal/mol)\n")
        f.write("=" * 50 + "\n")
        for i, result in enumerate(results, 1):
            f.write(f"{i}\t{result['folder']}\t{result['score']:.3f}\n")
    
    print(f"简明排名文件已保存到: {simple_filename}")
    
    # 显示摘要信息
    print("\n" + "=" * 80)
    print("统计摘要:")
    print("-" * 80)
    
    if results:
        scores = [r['score'] for r in results]
        min_score = min(scores)
        max_score = max(scores)
        avg_score = sum(scores) / len(scores)
        
        print(f"总结果数: {len(results)}")
        print(f"最佳对接分数: {min_score:.3f} kcal/mol")
        print(f"最差对接分数: {max_score:.3f} kcal/mol")
        print(f"平均对接分数: {avg_score:.3f} kcal/mol")
        
        # 显示前10个最佳结果
        print("\n前10个最佳对接结果:")
        print("-" * 60)
        print(f"{'排名':<6} {'文件夹名':<20} {'分数':<10} {'配体名':<20}")
        print("-" * 60)
        
        for i, result in enumerate(results[:10], 1):
            folder = result['folder']
            if len(folder) > 19:
                folder = folder[:17] + "..."
            
            ligand_name = result.get('ligand_name', 'N/A')
            if len(ligand_name) > 19:
                ligand_name = ligand_name[:17] + "..."
            
            print(f"{i:<6} {folder:<20} {result['score']:<10.3f} {ligand_name:<20}")
    
    print("\n" + "=" * 80)
    print("已生成的文件:")
    print(f"1. {csv_filename} - CSV格式，适合导入Excel")
    print(f"2. {txt_filename} - 完整报告，包含所有详细信息")
    print(f"3. {simple_filename} - 简明排名，只包含排名和分数")
    print("=" * 80)

if __name__ == "__main__":
    main()