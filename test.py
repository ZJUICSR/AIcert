import os

def count_lines_in_file(file_path):
    total_lines = 0
    comment_lines = 0
    effective_lines = 0

    with open(file_path, 'r') as f:
        in_multiline_comment = False
        for line in f:
            stripped_line = line.strip()
            total_lines += 1

            if stripped_line.startswith("'''") or stripped_line.startswith('"""'):
                comment_lines += 1
                in_multiline_comment = not in_multiline_comment
            elif in_multiline_comment:
                comment_lines += 1
            elif stripped_line.startswith('#'):
                comment_lines += 1
            elif stripped_line != '':
                effective_lines += 1

    return total_lines, comment_lines, effective_lines

def count_lines_in_project(project_path):
    total_lines = 0
    comment_lines = 0
    effective_lines = 0

    for foldername, subfolders, filenames in os.walk(project_path):
        for filename in filenames:
            if filename.endswith('.py'):
                file_total, file_comment, file_effective = count_lines_in_file(os.path.join(foldername, filename))
                total_lines += file_total
                comment_lines += file_comment
                effective_lines += file_effective

    return total_lines, comment_lines, effective_lines

project_path = './'  # 替换为你的项目路径
total, comments, effective = count_lines_in_project(project_path)
print(f'Total lines: {total}')
print(f'Comment lines: {comments}')
print(f'Effective lines: {effective}')