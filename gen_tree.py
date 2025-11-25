import os

def generate_project_structure(root_path, output_file):
    # 需要排除的文件夹和文件
    exclude_dirs = {'.git', '__pycache__', '.vscode', '.idea', 'wandb', '.hydra', 'robot_lab.egg-info', 'meshes'}
    exclude_files = {'.gitignore', '.DS_Store', 'project_directory.md', 'gen_tree.py', 'README.md'}

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# 项目主要目录结构\n\n")
        
        # 1. 先列出根目录下的文件
        f.write(f"## {root_path} (Root)\n\n")
        try:
            root_files = [fi for fi in os.listdir(root_path) if os.path.isfile(os.path.join(root_path, fi))]
            for fi in sorted(root_files):
                if fi not in exclude_files and not fi.startswith('.') and not fi.endswith('.pyc'):
                    f.write(f"- {fi}\n")
        except Exception as e:
            f.write(f"Error reading root: {e}\n")
        f.write("\n---\n\n")

        # 2. 遍历主要子目录
        target_dirs = ['source', 'scripts', 'logs', 'outputs']
        
        for target in target_dirs:
            target_path = os.path.join(root_path, target)
            if not os.path.exists(target_path):
                continue
                
            f.write(f"## {target_path}\n\n")
            
            for root, dirs, files in os.walk(target_path):
                # 过滤文件夹
                dirs[:] = [d for d in dirs if d not in exclude_dirs]
                
                # 计算缩进级别
                rel_path = os.path.relpath(root, target_path)
                if rel_path == '.':
                    level = 0
                else:
                    level = rel_path.count(os.sep) + 1
                
                indent = '  ' * level
                
                # 写入文件夹名（顶层除外，因为已在标题中）
                if rel_path != '.':
                    f.write(f"{indent}- {os.path.basename(root)}/\n")
                
                # 写入文件名
                sub_indent = '  ' * (level + 1)
                for file in sorted(files):
                    if file not in exclude_files and not file.endswith('.pyc'):
                        f.write(f"{sub_indent}- {file}\n")
            f.write("\n---\n\n")

if __name__ == "__main__":
    root = '/home/liu/Desktop/robot_lab'
    output = os.path.join(root, 'project_directory.md')
    generate_project_structure(root, output)
    print(f"✅ 目录结构已重新生成并写入: {output}")