#!/usr/bin/env python3
"""
AutoDock Vina 批量对接脚本 - 简化版
针对特定目录结构：vina、受体、配置在当前目录，配体在database中，输出到out
"""

import os
import sys
import subprocess
import time
from pathlib import Path
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class SimpleBatchDock:
    def __init__(self):
        """初始化，使用当前目录的固定结构"""
        self.current_dir = Path.cwd()
        self.vina_path = self.find_vina()
        self.receptor_file = self.current_dir / "receptor.pdbqt"
        self.config_file = self.current_dir / "config.txt"
        self.ligand_dir = self.current_dir / "database"
        self.output_dir = self.current_dir / "out"
        
        # 创建输出目录
        self.output_dir.mkdir(exist_ok=True)
        
        logger.info(f"工作目录: {self.current_dir}")
        logger.info(f"Vina路径: {self.vina_path}")
        logger.info(f"受体文件: {self.receptor_file}")
        logger.info(f"配置文件: {self.config_file}")
        logger.info(f"配体目录: {self.ligand_dir}")
        logger.info(f"输出目录: {self.output_dir}")
    
    def find_vina(self):
        """查找vina可执行文件"""
        possible_names = ["vina", "vina.exe", "./vina", "./vina.exe"]
        
        for name in possible_names:
            path = self.current_dir / name
            if path.exists():
                return str(path)
        
        # 如果在当前目录找不到，尝试在PATH中查找
        try:
            subprocess.run(["vina", "--version"], capture_output=True, check=True)
            return "vina"
        except:
            pass
        
        logger.warning("未找到vina可执行文件，尝试使用默认名称")
        return "vina"
    
    def get_ligand_files(self):
        """获取database目录中的所有配体文件"""
        ligand_files = []
        
        # 查找所有可能的配体格式
        extensions = ['.pdbqt', '.mol2', '.sdf', '.pdb']
        
        for ext in extensions:
            files = list(self.ligand_dir.glob(f"*{ext}"))
            ligand_files.extend(files)
        
        # 如果有子目录，也可以递归查找
        if not ligand_files:
            logger.info("在database根目录未找到配体文件，尝试递归查找...")
            for ext in extensions:
                files = list(self.ligand_dir.rglob(f"*{ext}"))
                ligand_files.extend(files)
        
        # 按文件名排序
        ligand_files.sort()
        
        logger.info(f"找到 {len(ligand_files)} 个配体文件")
        for i, f in enumerate(ligand_files[:10], 1):
            logger.info(f"  {i}. {f.name}")
        if len(ligand_files) > 10:
            logger.info(f"  ... 还有 {len(ligand_files)-10} 个文件")
        
        return ligand_files
    
    def convert_to_pdbqt(self, ligand_file, output_dir):
        """转换配体文件为PDBQT格式"""
        output_file = output_dir / f"{ligand_file.stem}.pdbqt"
        
        # 如果已经是pdbqt格式，直接复制
        if ligand_file.suffix.lower() == '.pdbqt':
            import shutil
            shutil.copy2(ligand_file, output_file)
            return output_file
        
        # 尝试使用Open Babel转换
        try:
            logger.info(f"转换: {ligand_file.name} -> {output_file.name}")
            cmd = ["obabel", str(ligand_file), "-O", str(output_file), "-h", "--gen3d"]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            if output_file.exists():
                return output_file
            else:
                raise RuntimeError("转换后文件不存在")
                
        except Exception as e:
            logger.error(f"转换失败 {ligand_file.name}: {e}")
            return None
    
    def run_single_docking(self, ligand_file, index, total):
        """运行单个对接"""
        # 创建配体特定的输出目录
        ligand_output_dir = self.output_dir / ligand_file.stem
        ligand_output_dir.mkdir(exist_ok=True)
        
        # 准备输出文件名
        output_file = ligand_output_dir / f"{ligand_file.stem}_docked.pdbqt"
        log_file = ligand_output_dir / f"{ligand_file.stem}_log.txt"
        
        # 如果配体不是PDBQT格式，先转换
        if ligand_file.suffix.lower() != '.pdbqt':
            pdbqt_file = self.convert_to_pdbqt(ligand_file, ligand_output_dir)
            if pdbqt_file is None:
                return None, "转换失败"
            ligand_file = pdbqt_file
        
        # 构建vina命令
        vina_cmd = [
            self.vina_path,
            "--receptor", str(self.receptor_file),
            "--ligand", str(ligand_file),
            "--config", str(self.config_file),
            "--out", str(output_file),
            #"--log", str(log_file),
            "--exhaustiveness", "8",
            "--num_modes", "10",
            "--energy_range", "4"
        ]
        
        try:
            logger.info(f"[{index}/{total}] 对接: {ligand_file.stem}")
            start_time = time.time()
            
            # 运行对接
            result = subprocess.run(
                vina_cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            elapsed_time = time.time() - start_time
            
            # 解析最佳结合亲和力
            affinity = self.parse_affinity(log_file)
            
            if output_file.exists():
                logger.info(f"  完成! 用时: {elapsed_time:.1f}秒, 亲和力: {affinity}")
                return output_file, affinity
            else:
                logger.error(f"  失败: 输出文件未生成")
                return None, "输出文件未生成"
                
        except subprocess.CalledProcessError as e:
            logger.error(f"  对接错误: {e.stderr[:200]}")
            return None, f"命令执行失败: {e.returncode}"
        except Exception as e:
            logger.error(f"  未知错误: {e}")
            return None, str(e)
    
    def parse_affinity(self, log_file):
        """从日志文件中解析结合亲和力"""
        try:
            if not log_file.exists():
                return "N/A"
            
            with open(log_file, 'r') as f:
                content = f.read()
            
            # 查找亲和力数值
            import re
            pattern = r"Affinity:\s*([-\d\.]+)\s*kcal/mol"
            match = re.search(pattern, content)
            
            if match:
                return f"{float(match.group(1)):.2f}"
            else:
                return "N/A"
                
        except Exception:
            return "N/A"
    
    def create_summary(self, results):
        """创建汇总报告"""
        summary_file = self.output_dir / "summary.txt"
        
        with open(summary_file, 'w') as f:
            f.write("AutoDock Vina 批量对接结果汇总\n")
            f.write("=" * 50 + "\n")
            f.write(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"受体文件: {self.receptor_file.name}\n")
            f.write(f"配置文件: {self.config_file.name}\n")
            f.write(f"配体数量: {len(results)}\n")
            f.write("=" * 50 + "\n\n")
            
            successful = 0
            for i, (ligand, success, output, affinity, error) in enumerate(results, 1):
                status = "成功" if success else f"失败: {error}"
                f.write(f"{i:3d}. {ligand:<30} {status:<20} 亲和力: {affinity}\n")
                if success:
                    successful += 1
            
            f.write("\n" + "=" * 50 + "\n")
            f.write(f"成功: {successful}/{len(results)} ({successful/len(results)*100:.1f}%)\n")
        
        logger.info(f"汇总报告已保存到: {summary_file}")
        return successful
    
    def run_batch_docking(self):
        """运行批量对接"""
        # 获取所有配体文件
        ligand_files = self.get_ligand_files()
        
        if not ligand_files:
            logger.error("在database目录中未找到任何配体文件")
            return
        
        logger.info(f"开始批量对接，共 {len(ligand_files)} 个配体")
        logger.info("-" * 50)
        
        results = []
        total_start_time = time.time()
        
        for i, ligand_file in enumerate(ligand_files, 1):
            output_file, affinity = self.run_single_docking(ligand_file, i, len(ligand_files))
            
            success = output_file is not None
            error_msg = affinity if not success else ""
            affinity_value = affinity if success else "N/A"
            
            results.append((ligand_file.name, success, output_file, affinity_value, error_msg))
            
            # 短暂暂停，避免系统负载过高
            time.sleep(0.1)
        
        total_time = time.time() - total_start_time
        
        # 创建汇总报告
        successful = self.create_summary(results)
        
        logger.info("-" * 50)
        logger.info(f"批量对接完成!")
        logger.info(f"总用时: {total_time:.1f}秒")
        logger.info(f"平均每个配体: {total_time/len(ligand_files):.1f}秒")
        logger.info(f"成功率: {successful}/{len(ligand_files)} ({successful/len(ligand_files)*100:.1f}%)")
        
        return results

def main():
    """主函数"""
    print("=" * 60)
    print("AutoDock Vina 批量对接脚本 - 简化版")
    print("=" * 60)
    print("目录结构要求:")
    print("  - vina可执行文件: 当前目录 (vina 或 vina.exe)")
    print("  - 受体文件: 当前目录/receptor.pdbqt")
    print("  - 配置文件: 当前目录/config.txt")
    print("  - 配体文件: 当前目录/database/ 目录下")
    print("  - 输出目录: 当前目录/out/ (自动创建)")
    print("=" * 60)
    
    # 检查必要文件
    required_files = ["receptor.pdbqt", "config.txt"]
    missing_files = []
    
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"错误: 缺少必要文件: {', '.join(missing_files)}")
        sys.exit(1)
    
    if not Path("database").exists():
        print("错误: database目录不存在")
        sys.exit(1)
    
    # 运行批量对接
    dock = SimpleBatchDock()
    
    try:
        results = dock.run_batch_docking()
        
        # 显示前几个结果
        if results:
            print("\n前10个配体对接结果:")
            print("-" * 60)
            successful_results = [r for r in results if r[1]]
            for i, (name, success, output, affinity, error) in enumerate(successful_results[:10], 1):
                print(f"{i:2d}. {name:<25} 亲和力: {affinity}")
        
    except KeyboardInterrupt:
        print("\n\n用户中断，正在退出...")
        sys.exit(0)
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()