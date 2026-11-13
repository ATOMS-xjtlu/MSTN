#!/usr/bin/env python3
"""
AutoDock Vina batch docking script - simplified version
Assumes a fixed layout: vina, receptor and config in the current directory, ligands in database/, output to out/.
"""

import os
import sys
import subprocess
import time
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class SimpleBatchDock:
    def __init__(self):
        """Initialise with the fixed layout of the current directory."""
        self.current_dir = Path.cwd()
        self.vina_path = self.find_vina()
        self.receptor_file = self.current_dir / "receptor.pdbqt"
        self.config_file = self.current_dir / "config.txt"
        self.ligand_dir = self.current_dir / "database"
        self.output_dir = self.current_dir / "out"
        
        self.output_dir.mkdir(exist_ok=True)
        
        logger.info(f"Working directory: {self.current_dir}")
        logger.info(f"Vina path: {self.vina_path}")
        logger.info(f"Receptor file: {self.receptor_file}")
        logger.info(f"Config file: {self.config_file}")
        logger.info(f"Ligand directory: {self.ligand_dir}")
        logger.info(f"Output directory: {self.output_dir}")
    
    def find_vina(self):
        """Locate the vina executable."""
        possible_names = ["vina", "vina.exe", "./vina", "./vina.exe"]
        
        for name in possible_names:
            path = self.current_dir / name
            if path.exists():
                return str(path)
        
        try:
            subprocess.run(["vina", "--version"], capture_output=True, check=True)
            return "vina"
        except:
            pass
        
        logger.warning("vina executable not found; trying the default name")
        return "vina"
    
    def get_ligand_files(self):
        """Collect all ligand files under database/."""
        ligand_files = []
        
        extensions = ['.pdbqt', '.mol2', '.sdf', '.pdb']
        
        for ext in extensions:
            files = list(self.ligand_dir.glob(f"*{ext}"))
            ligand_files.extend(files)
        
        if not ligand_files:
            logger.info("No ligand files in database/; searching recursively...")
            for ext in extensions:
                files = list(self.ligand_dir.rglob(f"*{ext}"))
                ligand_files.extend(files)
        
        ligand_files.sort()
        
        logger.info(f"Found {len(ligand_files)} ligand files")
        for i, f in enumerate(ligand_files[:10], 1):
            logger.info(f"  {i}. {f.name}")
        if len(ligand_files) > 10:
            logger.info(f"  ... and {len(ligand_files)-10} more")
        
        return ligand_files
    
    def convert_to_pdbqt(self, ligand_file, output_dir):
        """Convert a ligand file to PDBQT."""
        output_file = output_dir / f"{ligand_file.stem}.pdbqt"
        
        if ligand_file.suffix.lower() == '.pdbqt':
            import shutil
            shutil.copy2(ligand_file, output_file)
            return output_file
        
        try:
            logger.info(f"Converting: {ligand_file.name} -> {output_file.name}")
            cmd = ["obabel", str(ligand_file), "-O", str(output_file), "-h", "--gen3d"]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            if output_file.exists():
                return output_file
            else:
                raise RuntimeError("Converted file does not exist")
                
        except Exception as e:
            logger.error(f"Conversion failed for {ligand_file.name}: {e}")
            return None
    
    def run_single_docking(self, ligand_file, index, total):
        """Run one docking job."""
        ligand_output_dir = self.output_dir / ligand_file.stem
        ligand_output_dir.mkdir(exist_ok=True)
        
        output_file = ligand_output_dir / f"{ligand_file.stem}_docked.pdbqt"
        log_file = ligand_output_dir / f"{ligand_file.stem}_log.txt"
        
        if ligand_file.suffix.lower() != '.pdbqt':
            pdbqt_file = self.convert_to_pdbqt(ligand_file, ligand_output_dir)
            if pdbqt_file is None:
                return None, "conversion failed"
            ligand_file = pdbqt_file
        
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
            logger.info(f"[{index}/{total}] Docking: {ligand_file.stem}")
            start_time = time.time()
            
            result = subprocess.run(
                vina_cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            elapsed_time = time.time() - start_time
            
            affinity = self.parse_affinity(log_file)
            
            if output_file.exists():
                logger.info(f"  Done. Time: {elapsed_time:.1f} s, affinity: {affinity}")
                return output_file, affinity
            else:
                logger.error(f"  Failed: output file was not created")
                return None, "output file was not created"
                
        except subprocess.CalledProcessError as e:
            logger.error(f"  Docking error: {e.stderr[:200]}")
            return None, f"command failed: {e.returncode}"
        except Exception as e:
            logger.error(f"  Unknown error: {e}")
            return None, str(e)
    
    def parse_affinity(self, log_file):
        """Parse the binding affinity from the log file."""
        try:
            if not log_file.exists():
                return "N/A"
            
            with open(log_file, 'r') as f:
                content = f.read()
            
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
        """Write the summary report."""
        summary_file = self.output_dir / "summary.txt"
        
        with open(summary_file, 'w') as f:
            f.write("AutoDock Vina batch docking summary\n")
            f.write("=" * 50 + "\n")
            f.write(f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Receptor: {self.receptor_file.name}\n")
            f.write(f"Config: {self.config_file.name}\n")
            f.write(f"Ligands: {len(results)}\n")
            f.write("=" * 50 + "\n\n")
            
            successful = 0
            for i, (ligand, success, output, affinity, error) in enumerate(results, 1):
                status = "OK" if success else f"FAILED: {error}"
                f.write(f"{i:3d}. {ligand:<30} {status:<20} Affinity: {affinity}\n")
                if success:
                    successful += 1
            
            f.write("\n" + "=" * 50 + "\n")
            f.write(f"Succeeded: {successful}/{len(results)} ({successful/len(results)*100:.1f}%)\n")
        
        logger.info(f"Summary report saved to: {summary_file}")
        return successful
    
    def run_batch_docking(self):
        """Run batch docking."""
        ligand_files = self.get_ligand_files()
        
        if not ligand_files:
            logger.error("No ligand files found in database/")
            return
        
        logger.info(f"Starting batch docking for {len(ligand_files)} ligands")
        logger.info("-" * 50)
        
        results = []
        total_start_time = time.time()
        
        for i, ligand_file in enumerate(ligand_files, 1):
            output_file, affinity = self.run_single_docking(ligand_file, i, len(ligand_files))
            
            success = output_file is not None
            error_msg = affinity if not success else ""
            affinity_value = affinity if success else "N/A"
            
            results.append((ligand_file.name, success, output_file, affinity_value, error_msg))
            
            time.sleep(0.1)
        
        total_time = time.time() - total_start_time
        
        successful = self.create_summary(results)
        
        logger.info("-" * 50)
        logger.info(f"Batch docking finished!")
        logger.info(f"Total time: {total_time:.1f} s")
        logger.info(f"Average per ligand: {total_time/len(ligand_files):.1f} s")
        logger.info(f"Success rate: {successful}/{len(ligand_files)} ({successful/len(ligand_files)*100:.1f}%)")
        
        return results

def main():
    """Main entry point."""
    print("=" * 60)
    print("AutoDock Vina batch docking script - simplified version")
    print("=" * 60)
    print("Required directory layout:")
    print("  - vina executable: current directory (vina or vina.exe)")
    print("  - receptor: current directory/receptor.pdbqt")
    print("  - config: current directory/config.txt")
    print("  - ligands: current directory/database/")
    print("  - output: current directory/out/ (created automatically)")
    print("=" * 60)
    
    required_files = ["receptor.pdbqt", "config.txt"]
    missing_files = []
    
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"Error: missing required files: {', '.join(missing_files)}")
        sys.exit(1)
    
    if not Path("database").exists():
        print("Error: database directory does not exist")
        sys.exit(1)
    
    dock = SimpleBatchDock()
    
    try:
        results = dock.run_batch_docking()
        
        if results:
            print("\nTop 10 docking results:")
            print("-" * 60)
            successful_results = [r for r in results if r[1]]
            for i, (name, success, output, affinity, error) in enumerate(successful_results[:10], 1):
                print(f"{i:2d}. {name:<25} Affinity: {affinity}")
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user; exiting...")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()