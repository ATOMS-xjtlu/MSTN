#!/usr/bin/env python3
"""
AutoDock Vina result summary and ranking script
Extract docking scores from every result under out/, rank them and write complete record files.
"""

import os
import re
import csv
import time
from pathlib import Path
from datetime import datetime

def extract_vina_score_from_file(file_path: Path) -> float:
    """
    Extract the Vina docking score from a PDBQT file.
    
    Parameters:
        file_path: path to the PDBQT file
        
    Returns:
        Docking score (float), or None if not found.
    """
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if "VINA RESULT:" in line:
                    match = re.search(r'[-+]?\d*\.\d+', line)
                    if match:
                        return float(match.group())
        return None
    except Exception as e:
        print(f"  File read error: {e}")
        return None

def extract_additional_info_from_file(file_path: Path) -> dict:
    """
    Extract additional information from a PDBQT file.
    
    Parameters:
        file_path: path to the PDBQT file
        
    Returns:
        Dictionary with the additional information.
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
            content = f.read(2000)
            
            match = re.search(r'INTER \+ INTRA:\s*([-+]?\d*\.\d+)', content)
            if match:
                info['inter_intra'] = float(match.group(1))
            
            match = re.search(r'INTER:\s*([-+]?\d*\.\d+)', content)
            if match:
                info['inter'] = float(match.group(1))
            
            match = re.search(r'INTRA:\s*([-+]?\d*\.\d+)', content)
            if match:
                info['intra'] = float(match.group(1))
            
            match = re.search(r'UNBOUND:\s*([-+]?\d*\.\d+)', content)
            if match:
                info['unbound'] = float(match.group(1))
            
            match = re.search(r'Name\s*=\s*(\S+)', content)
            if match:
                info['ligand_name'] = match.group(1)
            
            match = re.search(r'(\d+)\s+active torsions:', content)
            if match:
                info['active_torsions'] = int(match.group(1))
    
    except Exception as e:
        print(f"  Error extracting additional information: {e}")
    
    return info

def generate_complete_report(results, output_file="vina_complete_report.txt"):
    """
    Write the complete report with all results and statistics.
    
    Parameters:
        results: sorted list of results
        output_file: output file name
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 120 + "\n")
        f.write("AUTODOCK VINA DOCKING RESULTS - FULL REPORT\n")
        f.write("=" * 120 + "\n")
        f.write(f"Generated at: {timestamp}\n")
        f.write(f"Total results: {len(results)}\n")
        f.write("=" * 120 + "\n\n")
        
        f.write("Summary statistics:\n")
        f.write("-" * 120 + "\n")
        
        if results:
            scores = [r['score'] for r in results]
            min_score = min(scores)
            max_score = max(scores)
            avg_score = sum(scores) / len(scores)
            
            f.write(f"Best score: {min_score:.3f} kcal/mol\n")
            f.write(f"Worst score: {max_score:.3f} kcal/mol\n")
            f.write(f"Mean score: {avg_score:.3f} kcal/mol\n")
            f.write(f"Median score: {sorted(scores)[len(scores)//2]:.3f} kcal/mol\n")
            
            f.write("\nScore distribution:\n")
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
                f.write(f"  {label:<12}: {count:>3} ({percentage:>5.1f}%)\n")
        
        f.write("\n" + "=" * 120 + "\n")
        f.write("Full result list (sorted by score, best to worst):\n")
        f.write("=" * 120 + "\n\n")
        
        header = f"{'Rank':<6} {'Folder':<20} {'Score':<12} {'INTER':<10} {'INTRA':<10} {'Ligand':<25} {'Torsions':<8} {'File':<20}"
        f.write(header + "\n")
        f.write("-" * 120 + "\n")
        
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
        f.write("End of file\n")
    
    print(f"Full report saved to: {output_file}")

def main():
    print("=" * 80)
    print("AutoDock Vina result summary and ranking tool")
    print("=" * 80)
    print("Scanning out/ ...")
    
    results = []
    out_dir = Path("out")
    
    if not out_dir.exists():
        print("Error: out/ does not exist")
        print("Make sure the script and out/ are in the same directory")
        return
    
    folders = sorted([f for f in out_dir.iterdir() if f.is_dir()])
    
    if not folders:
        print("Error: out/ contains no subdirectories")
        return
    
    print(f"Found {len(folders)} subdirectories")
    
    start_time = time.time()
    
    for i, folder in enumerate(folders, 1):
        print(f"[{i:3d}/{len(folders)}] Processing: {folder.name}")
        
        pdbqt_files = list(folder.glob("*.pdbqt"))
        
        if not pdbqt_files:
            print(f"  Warning: no .pdbqt file found")
            continue
        
        pdbqt_file = pdbqt_files[0]
        
        score = extract_vina_score_from_file(pdbqt_file)
        
        if score is None:
            print(f"  Warning: no docking score found")
            continue
        
        extra_info = extract_additional_info_from_file(pdbqt_file)
        
        result = {
            'folder': folder.name,
            'score': score,
            'file': pdbqt_file.name,
            'full_path': str(pdbqt_file)
        }
        
        result.update(extra_info)
        
        results.append(result)
        print(f"  OK: score = {score:.3f}")
    
    if not results:
        print("No valid docking results found")
        return
    
    elapsed_time = time.time() - start_time
    print(f"\nDone. Elapsed: {elapsed_time:.2f} s")
    print(f"Extracted {len(results)} docking results")
    
    print("Sorting by docking score...")
    results.sort(key=lambda x: x['score'])
    
    print("Writing CSV file...")
    csv_filename = "vina_results_sorted.csv"
    with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
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
    
    print(f"CSV saved to: {csv_filename}")
    
    print("Writing full report...")
    txt_filename = "vina_results_complete_report.txt"
    generate_complete_report(results, txt_filename)
    
    print("Writing concise ranking file...")
    simple_filename = "vina_results_ranking.txt"
    with open(simple_filename, 'w', encoding='utf-8') as f:
        f.write("Rank\tFolder\tScore(kcal/mol)\n")
        f.write("=" * 50 + "\n")
        for i, result in enumerate(results, 1):
            f.write(f"{i}\t{result['folder']}\t{result['score']:.3f}\n")
    
    print(f"Concise ranking saved to: {simple_filename}")
    
    print("\n" + "=" * 80)
    print("Summary statistics:")
    print("-" * 80)
    
    if results:
        scores = [r['score'] for r in results]
        min_score = min(scores)
        max_score = max(scores)
        avg_score = sum(scores) / len(scores)
        
        print(f"Total results: {len(results)}")
        print(f"Best score: {min_score:.3f} kcal/mol")
        print(f"Worst score: {max_score:.3f} kcal/mol")
        print(f"Mean score: {avg_score:.3f} kcal/mol")
        
        print("\nTop 10 docking results:")
        print("-" * 60)
        print(f"{'Rank':<6} {'Folder':<20} {'Score':<10} {'Ligand':<20}")
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
    print("Generated files:")
    print(f"1. {csv_filename} - CSV format, suitable for Excel")
    print(f"2. {txt_filename} - full report with all details")
    print(f"3. {simple_filename} - concise ranking with rank and score only")
    print("=" * 80)

if __name__ == "__main__":
    main()