import os
import json
import subprocess
import tempfile
import shutil
import logging
from pathlib import Path
import time
import glob

# ====================== 配置区域 =======================
JOERN_PATH = "/home/huoguoyuxi/name/joern-cli-new"
DATA_BASE = "/home/huoguoyuxi/name/yuxi/data"
GRAPH_BASE = "/home/huoguoyuxi/name/yuxi/graph"
LOG_FILE = "/home/huoguoyuxi/name/yuxi/code/dataprocess.log"

# 批处理配置
BATCH_SIZE = 500  # 每批处理的函数数量
MAX_FUNCTION_SIZE = 50000  # 最大函数字符数限制
MAX_LINES = 2000  # 最大行数限制

DATASETS = {
    # "FFMPeg": ["test_cdata.jsonl", "train_cdata.jsonl", "valid_cdata.jsonl"],
    # "Reveal": ["test_cdata.jsonl", "train_cdata.jsonl", "valid_cdata.jsonl"], 
    # "SVulD": ["bigvul_dataset.jsonl"],
    # "Devign": ["devign_dataset.jsonl"],
    # "DiverseVul": ["diversevul_dataset.jsonl"],
    "VDISC": ["vdisc_dataset.jsonl"]
}

# 更新图类型配置
GRAPH_TYPES = {
    "ast": "AST",
    "cfg": "CFG", 
    "ddg": "DDG",  # 改为 DDG（数据依赖图）
    "cpg": "CPG"
}


def setup_logging():
    """设置日志配置"""
    log_dir = os.path.dirname(LOG_FILE)
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def create_directories():
    """创建所有必要的目录结构"""
    for dataset in DATASETS.keys():
        dataset_path = os.path.join(GRAPH_BASE, dataset)
        for folder in GRAPH_TYPES.values():
            os.makedirs(os.path.join(dataset_path, folder), exist_ok=True)

def run_command(cmd, cwd=None, timeout=600):
    """执行命令，带超时控制和详细输出记录"""
    logger = logging.getLogger(__name__)
    logger.debug(f"执行命令: {cmd}")
    
    try:
        # 使用实时输出捕获
        process = subprocess.Popen(
            cmd,
            cwd=cwd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # 实时读取输出
        stdout_lines = []
        stderr_lines = []
        
        # 等待进程完成，同时收集输出
        try:
            stdout, stderr = process.communicate(timeout=timeout)
            stdout_lines.append(stdout)
            stderr_lines.append(stderr)
        except subprocess.TimeoutExpired:
            process.kill()
            logger.error(f"命令执行超时: {cmd}")
            return -1, "", "Timeout"
        
        full_stdout = ''.join(stdout_lines)
        full_stderr = ''.join(stderr_lines)
        
        return process.returncode, full_stdout, full_stderr
        
    except Exception as e:
        logger.error(f"命令执行异常: {e}")
        return -1, "", str(e)

def clean_joern_workspace():
    """清理 Joern 工作空间"""
    workspace_path = os.path.join(JOERN_PATH, "workspace")
    if os.path.exists(workspace_path):
        shutil.rmtree(workspace_path, ignore_errors=True)

def check_function_size(func_code):
    """检查函数是否过大，无法处理"""
    if not func_code:
        return False, "函数代码为空"
    
    if len(func_code) > MAX_FUNCTION_SIZE:
        return False, f"函数过大 ({len(func_code)} 字符 > {MAX_FUNCTION_SIZE})"
    
    line_count = func_code.count('\n') + 1
    if line_count > MAX_LINES:
        return False, f"函数行数过多 ({line_count} 行 > {MAX_LINES})"
    
    return True, "OK"

def check_files_exist(dataset, target, idx):
    """检查所有图文件是否已存在"""
    for repr_type, folder in GRAPH_TYPES.items():
        filename = f"{target}_{idx}.{repr_type}.dot"
        file_path = os.path.join(GRAPH_BASE, dataset, folder, filename)
        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            return False
    return True

def collect_batch_functions(dataset, jsonl_file, start_line=0):
    """收集一批函数数据，返回 (functions_data, next_start_line, is_finished)"""
    logger = logging.getLogger(__name__)
    file_path = os.path.join(DATA_BASE, dataset, jsonl_file)
    
    if not os.path.exists(file_path):
        return [], 0, True
    
    functions_data = []
    current_line = 0
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line_num <= start_line:
                continue
            
            current_line = line_num
            
            try:
                data = json.loads(line.strip())
                idx = data.get('idx', line_num)
                func_code = data.get('func', '')
                target = data.get('target', 0)
                
                if not func_code or not isinstance(func_code, str):
                    continue
                    
                can_process, _ = check_function_size(func_code)
                if not can_process:
                    continue
                
                if check_files_exist(dataset, target, idx):
                    continue
                
                functions_data.append({
                    'dataset': dataset,
                    'idx': idx,
                    'target': target,
                    'func_code': func_code,
                    'line_num': line_num
                })
                
                if len(functions_data) >= BATCH_SIZE:
                    return functions_data, current_line, False
                    
            except Exception as e:
                logger.error(f"处理第{line_num}行时出错: {e}")
                continue
    
    return functions_data, current_line, True

def create_batch_temp_files(functions_data, temp_dir):
    """创建批量临时 C 文件"""
    logger = logging.getLogger(__name__)
    file_mapping = {}
    
    for func_data in functions_data:
        dataset = func_data['dataset']
        idx = func_data['idx']
        target = func_data['target']
        func_code = func_data['func_code']
        
        # 使用新的命名格式：dataset_target_idx.c
        filename = f"{dataset}_{target}_{idx}.c"
        file_path = os.path.join(temp_dir, filename)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(func_code)
        
        file_mapping[filename] = {
            'dataset': dataset,
            'idx': idx,
            'target': target
        }
    
    logger.info(f"创建了 {len(file_mapping)} 个临时 C 文件")
    return file_mapping

def create_joern_script(temp_dir):
    """创建改进的 Joern 脚本，增加更详细的调试输出"""
    script_content = f'''
// 改进版按文件生成图的批处理脚本 - 带详细调试输出
println("=== Joern 脚本开始执行 ===")
println(s"临时目录: {temp_dir}")

try {{
  println("导入代码...")
  importCode("{temp_dir}")
  println("✓ 代码导入成功")
}} catch {{
  case e: Exception =>
    println(s"✗ 代码导入失败: ${{e.getMessage}}")
    e.printStackTrace()
    sys.exit(1)
}}

// 检查导入的文件数量
val fileCount = cpg.file.l.size
println(s"导入的文件数量: $fileCount")

if (fileCount == 0) {{
  println("⚠ 没有导入任何文件")
  sys.exit(1)
}}

// 创建输出目录
import java.nio.file.{{Files, Paths}}
val outputBase = "{temp_dir}/joern_output"
Array("ast", "cfg", "ddg", "cpg").foreach {{ dir =>
  val dirPath = Paths.get(s"$outputBase/$dir")
  Files.createDirectories(dirPath)
  println(s"创建输出目录: $dirPath")
}}

println("=== 开始运行数据流分析 ===")
try {{
  run.ossdataflow
  println("✓ 数据流分析完成")
}} catch {{
  case e: Exception =>
    println(s"✗ 数据流分析失败: ${{e.getMessage}}")
    e.printStackTrace()
    // 不要因为数据流分析失败就退出，继续尝试生成其他图
}}

println("=== 开始按文件生成图 ===")

// 获取所有文件，按文件分组处理
val files = cpg.file.l
println(s"开始处理 ${{files.size}} 个文件")

var processedCount = 0
var successCount = 0
var errorCount = 0

files.foreach {{ file =>
  val fileName = file.name.split("[/\\\\\\\\]").last.replace(".c", "")
  println(s"\\n[${{processedCount + 1}}/${{files.size}}] 处理文件: $fileName.c")
  processedCount += 1
  
  try {{
    // 获取该文件中的所有真实方法（排除全局和内置方法）
    val fileMethods = cpg.method.filename(file.name).isExternal(false).l.filter(_.name != "<global>")
    val globalMethods = cpg.method.filename(file.name).name("<global>").l
    
    println(s"  文件方法数: ${{fileMethods.size}}, 全局方法数: ${{globalMethods.size}}")
    
    val targetMethod = if (fileMethods.nonEmpty) {{
      val method = fileMethods.maxBy(m => m.lineNumber.getOrElse(0) -> m.ast.l.size)
      println(s"  选择方法: ${{method.name}} (行号: ${{method.lineNumber.getOrElse("N/A")}}, AST节点: ${{method.ast.l.size}})")
      method
    }} else if (globalMethods.nonEmpty) {{
      val method = globalMethods.head
      println(s"  使用全局方法 (AST节点: ${{method.ast.l.size}})")
      method
    }} else {{
      println(s"  ⚠ 文件中没有找到任何方法")
      null
    }}
    
    if (targetMethod != null) {{
      var fileSuccess = true
      
      // 生成AST图
      try {{
        val astDot = targetMethod.ast.dotAst.l.headOption
        if (astDot.isDefined) {{
          val astPath = s"$outputBase/ast/${{fileName}}.dot"
          astDot.get |> astPath
          println(s"  ✓ AST图已生成: $astPath")
        }} else {{
          println(s"  ⚠ AST为空")
          fileSuccess = false
        }}
      }} catch {{
        case e: Exception =>
          println(s"  ✗ AST生成失败: ${{e.getMessage}}")
          fileSuccess = false
      }}
      
      // 生成CFG图
      try {{
        val cfgDot = targetMethod.dotCfg.l.headOption
        if (cfgDot.isDefined) {{
          val cfgPath = s"$outputBase/cfg/${{fileName}}.dot"
          cfgDot.get |> cfgPath
          println(s"  ✓ CFG图已生成: $cfgPath")
        }} else {{
          println(s"  ⚠ CFG为空")
        }}
      }} catch {{
        case e: Exception =>
          println(s"  ✗ CFG生成失败: ${{e.getMessage}}")
      }}
      
      // 生成DDG图（数据依赖图）
      try {{
        val ddgDot = targetMethod.dotDdg.l.headOption
        if (ddgDot.isDefined) {{
          val ddgPath = s"$outputBase/ddg/${{fileName}}.dot"
          ddgDot.get |> ddgPath
          println(s"  ✓ DDG图已生成: $ddgPath")
        }} else {{
          println(s"  ⚠ DDG为空 (可能需要数据流分析)")
        }}
      }} catch {{
        case e: Exception =>
          println(s"  ✗ DDG生成失败: ${{e.getMessage}}")
      }}
      
      // 生成CPG图
      try {{
        val cpgDot = targetMethod.dotCpg14.l.headOption
        if (cpgDot.isDefined) {{
          val cpgPath = s"$outputBase/cpg/${{fileName}}.dot"
          cpgDot.get |> cpgPath
          println(s"  ✓ CPG图已生成: $cpgPath")
        }} else {{
          println(s"  ⚠ CPG为空")
        }}
      }} catch {{
        case e: Exception =>
          println(s"  ✗ CPG生成失败: ${{e.getMessage}}")
      }}
      
      if (fileSuccess) successCount += 1 else errorCount += 1
      
    }} else {{
      println(s"  ✗ 跳过文件 $fileName.c (没有有效方法)")
      errorCount += 1
    }}
    
  }} catch {{
    case e: Exception =>
      println(s"  ✗ 文件处理异常: ${{e.getMessage}}")
      e.printStackTrace()
      errorCount += 1
  }}
}}

println(s"\\n=== 处理完成 ===")
println(s"总文件数: ${{files.size}}")
println(s"处理成功: $successCount")
println(s"处理失败: $errorCount")

// 统计生成的图文件
Array("ast", "cfg", "ddg", "cpg").foreach {{ graphType =>
  val graphDir = s"$outputBase/$graphType"
  val dotFiles = java.nio.file.Files.list(java.nio.file.Paths.get(graphDir))
    .iterator().asScala.filter(_.toString.endsWith(".dot")).size
  println(s"$graphType 图文件数量: $dotFiles")
}}

println("=== Joern 脚本执行完毕 ===")
'''
    
    script_path = os.path.join(temp_dir, "batch_process.sc")
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    return script_path

def process_batch_with_joern(temp_dir, file_mapping):
    """使用 Joern 批量处理一批函数（使用新脚本）"""
    logger = logging.getLogger(__name__)
    
    clean_joern_workspace()
    
    # 创建 Joern 脚本
    script_path = create_joern_script(temp_dir)
    
    logger.info("开始使用 Joern 脚本批量处理...")
    
    # 运行 Joern 脚本
    cmd = f"./joern --script '{script_path}'"
    returncode, stdout, stderr = run_command(cmd, cwd=JOERN_PATH, timeout=1200)
    
    # 记录详细的输出
    logger.info("=" * 50 + " JOERN OUTPUT " + "=" * 50)
    if stdout:
        logger.info("STDOUT:")
        for line in stdout.split('\n'):
            if line.strip():
                logger.info(f"  {line}")
    
    if stderr:
        logger.info("STDERR:")
        for line in stderr.split('\n'):
            if line.strip():
                logger.info(f"  {line}")
    logger.info("=" * 110)
    
    if returncode != 0:
        logger.error(f"Joern 脚本执行失败，返回码: {returncode}")
        return False
    
    logger.info("Joern 脚本执行成功")
    
    # 检查输出目录是否存在
    joern_output_dir = os.path.join(temp_dir, "joern_output")
    if not os.path.exists(joern_output_dir):
        logger.error("Joern 输出目录不存在")
        return False
    
    # 检查是否有输出文件
    has_output = False
    for graph_type in GRAPH_TYPES.keys():
        graph_dir = os.path.join(joern_output_dir, graph_type)
        if os.path.exists(graph_dir):
            files = os.listdir(graph_dir)
            file_count = len([f for f in files if f.endswith('.dot')])
            logger.info(f"{graph_type} 目录包含 {file_count} 个 dot 文件")
            if file_count > 0:
                has_output = True
        else:
            logger.warning(f"{graph_type} 输出目录不存在")
    
    if not has_output:
        logger.warning("没有生成任何图文件")
        return False
    
    return True

def organize_output_files(temp_dir, file_mapping):
    """整理输出文件到正确位置"""
    logger = logging.getLogger(__name__)
    moved_count = 0
    
    joern_output_dir = os.path.join(temp_dir, "joern_output")
    
    if not os.path.exists(joern_output_dir):
        logger.error("Joern 输出目录不存在")
        return 0
    
    for graph_type, folder_name in GRAPH_TYPES.items():
        graph_output_dir = os.path.join(joern_output_dir, graph_type)
        
        if not os.path.exists(graph_output_dir):
            logger.warning(f"图类型输出目录不存在: {graph_output_dir}")
            continue
        
        dot_files = glob.glob(os.path.join(graph_output_dir, "*.dot"))
        logger.info(f"在 {graph_type} 中找到 {len(dot_files)} 个dot文件")
        
        # 记录所有找到的文件名，用于调试
        if dot_files:
            logger.debug(f"{graph_type} 文件列表: {[os.path.basename(f) for f in dot_files[:10]]}")  # 只显示前10个
        
        for dot_file in dot_files:
            try:
                # 从dot文件名推断原始信息
                dot_filename = os.path.basename(dot_file)
                base_name = dot_filename.replace('.dot', '')
                
                # 跳过特殊文件
                if base_name in ['<includes>', '<unknown>']:
                    logger.warning(f"跳过特殊文件: {dot_filename}")
                    continue
                
                # 查找匹配的映射信息
                matching_mapping = None
                for c_filename, mapping_info in file_mapping.items():
                    expected_name = f"{mapping_info['dataset']}_{mapping_info['target']}_{mapping_info['idx']}"
                    if base_name == expected_name:
                        matching_mapping = mapping_info
                        break
                
                if matching_mapping is None:
                    logger.warning(f"无法找到匹配的映射信息: {dot_filename}")
                    # 记录可能的匹配选项
                    logger.debug(f"可能的映射: {list(file_mapping.keys())[:5]}")  # 显示前5个作为参考
                    continue
                
                # 构造目标文件名和路径
                dataset = matching_mapping['dataset']
                idx = matching_mapping['idx']
                target = matching_mapping['target']
                
                new_filename = f"{target}_{idx}.{graph_type}.dot"
                dst_path = os.path.join(GRAPH_BASE, dataset, folder_name, new_filename)
                
                # 移动文件
                shutil.move(dot_file, dst_path)
                moved_count += 1
                logger.debug(f"移动文件: {dot_filename} -> {new_filename}")
                
            except Exception as e:
                logger.error(f"移动文件失败 {dot_file}: {e}")
    
    logger.info(f"成功移动 {moved_count} 个图文件")
    return int(moved_count/4) if moved_count > 0 else 0

def process_batch(functions_data, batch_num):
    """处理一批函数"""
    logger = logging.getLogger(__name__)
    logger.info(f"开始处理批次 {batch_num}，包含 {len(functions_data)} 个函数")
    
    if not functions_data:
        return 0, 0
    
    temp_dir = tempfile.mkdtemp(prefix=f"joern_batch_{batch_num}_")
    logger.info(f"批次 {batch_num} 临时目录: {temp_dir}")
    
    try:
        file_mapping = create_batch_temp_files(functions_data, temp_dir)
        
        if not file_mapping:
            logger.warning(f"批次 {batch_num} 没有有效的函数文件")
            return 0, len(functions_data)
        
        # 记录一些样本文件映射，用于调试
        sample_mappings = list(file_mapping.items())[:3]
        logger.debug(f"样本文件映射: {sample_mappings}")
        
        if not process_batch_with_joern(temp_dir, file_mapping):
            logger.error(f"批次 {batch_num} Joern 处理失败")
            return 0, len(functions_data)
        
        moved_count = organize_output_files(temp_dir, file_mapping)
        
        logger.info(f"批次 {batch_num} 完成: 成功处理 {moved_count} 个函数，失败 {len(functions_data) - moved_count} 个")
        return moved_count, len(functions_data) - moved_count
        
    except Exception as e:
        logger.error(f"批次 {batch_num} 处理异常: {e}")
        import traceback
        logger.error(f"异常详情: {traceback.format_exc()}")
        return 0, len(functions_data)
        
    finally:
        try:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                logger.debug(f"清理临时目录: {temp_dir}")
        except Exception as e:
            logger.warning(f"清理临时目录失败: {e}")

def process_jsonl_file_in_batches(dataset, jsonl_file):
    """分批处理 JSONL 文件"""
    logger = logging.getLogger(__name__)
    logger.info(f"开始分批处理: {dataset}/{jsonl_file}")
    
    start_line = 0
    batch_num = 1
    total_success = 0
    total_error = 0
    
    while True:
        functions_data, next_start_line, is_finished = collect_batch_functions(
            dataset, jsonl_file, start_line
        )
        
        if not functions_data:
            if is_finished:
                break
            else:
                start_line = next_start_line
                continue
        
        success_count, error_count = process_batch(functions_data, batch_num)
        total_success += success_count
        total_error += error_count
        
        logger.info(f"{dataset}/{jsonl_file} 批次 {batch_num} 完成: 成功={success_count}, 失败={error_count}")
        
        if is_finished:
            break
        
        start_line = next_start_line
        batch_num += 1
        
        # 添加批次间的短暂休息
        time.sleep(2)
    
    logger.info(f"{dataset}/{jsonl_file} 全部完成: 总成功={total_success}, 总失败={total_error}, 共处理 {batch_num-1} 批")

def main():
    """主函数"""
    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info(f"开始批量处理代码图生成 (批次大小: {BATCH_SIZE})")
    logger.info("=" * 80)
    
    create_directories()
    
    # 检查 Joern 可执行文件
    joern_executable = os.path.join(JOERN_PATH, "joern")
    if not os.path.exists(joern_executable):
        logger.error(f"joern 可执行文件不存在: {joern_executable}")
        return
    
    start_time = time.time()
    
    for dataset, jsonl_files in DATASETS.items():
        logger.info(f"开始处理数据集: {dataset}")
        
        for jsonl_file in jsonl_files:
            process_jsonl_file_in_batches(dataset, jsonl_file)
    
    elapsed_time = time.time() - start_time
    logger.info("=" * 80)
    logger.info(f"所有数据集处理完成！总耗时: {elapsed_time:.2f} 秒")
    logger.info(f"图文件保存在: {GRAPH_BASE}")
    logger.info("=" * 80)

if __name__ == "__main__":
    logger = setup_logging()
    
    if not os.path.exists(JOERN_PATH):
        logger.error(f"Joern路径不存在: {JOERN_PATH}")
        exit(1)
    
    logger.info(f"开始处理，批次大小: {BATCH_SIZE}")
    main()
