import os
import re
import sys
import pkgutil
import importlib.util

def find_imports(directory):
    import_set = set()
    import_pattern = re.compile(r'^\s*(?:import|from)\s+([\w\.]+)')

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                full_path = os.path.join(root, file)
                with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        match = import_pattern.match(line)
                        if match:
                            module = match.group(1).split('.')[0]  # 顶层模块
                            import_set.add(module)
    return sorted(import_set)

def is_third_party_package(package_name):
    # 尝试查找模块信息
    spec = importlib.util.find_spec(package_name)
    if spec is None or spec.origin is None:
        return False
    # 判断是否在 site-packages 目录中
    return 'site-packages' in spec.origin or 'dist-packages' in spec.origin

def generate_requirements(directory, output_file="requirements.txt"):
    all_imports = find_imports(directory)
    third_party = [pkg for pkg in all_imports if is_third_party_package(pkg)]

    with open(output_file, 'w') as f:
        for pkg in sorted(third_party):
            f.write(f"{pkg}\n")

    print(f"✅ requirements.txt generated with {len(third_party)} packages.")

# 修改成你的项目路径
project_path = "/home/YaoGuo/code/EEG-AugBench/"
generate_requirements(project_path)
