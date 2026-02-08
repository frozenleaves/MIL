#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从医疗病例txt文件中提取指定字段（年龄、性别、主诉、现病史、既往史、评估、体格检查、辅助检查），
生成同名 -mask.txt 文件。
"""

import os
import re
import glob

BASE_DIR = "/media/codingma/LLM/lcx/Medical_Info_Classification/datasets"

# 定义需要提取的字段和终止字段的正则模式
# 字段名中可能有空格，如 "主    诉" "现 病 史" 等
FIELD_PATTERNS = {
    '主诉': re.compile(r'^主\s*诉\s*[：:]'),
    '现病史': re.compile(r'^现\s*病\s*史\s*[：:]'),
    '既往史': re.compile(r'^既\s*往\s*史\s*[：:]'),
    '评估': re.compile(r'^评\s*估\s*[：:]'),
    '体格检查': re.compile(r'^体\s*格\s*检\s*查\s*[：:]'),
    '辅助检查': re.compile(r'^辅\s*助\s*检\s*查\s*[：:]'),
}

# 终止字段：遇到这些字段名表示目标字段已结束
STOP_PATTERNS = {
    '诊断': re.compile(r'^诊\s*断\s*[：:]'),
    '治疗计划': re.compile(r'^治\s*疗\s*计\s*划\s*[：:]'),
    '处置': re.compile(r'^处\s*置\s*[：:]'),
    '医生签名': re.compile(r'^医\s*生\s*签\s*名\s*[：:]'),
}

# 所有字段模式（包括需要提取的和终止的），用于判断一行是否是新字段的开始
ALL_FIELD_PATTERNS = {}
ALL_FIELD_PATTERNS.update(FIELD_PATTERNS)
ALL_FIELD_PATTERNS.update(STOP_PATTERNS)
# 也加入姓名行等
ALL_FIELD_PATTERNS['姓名'] = re.compile(r'^姓\s*名\s*[：:]')
ALL_FIELD_PATTERNS['主索引号'] = re.compile(r'^主\s*索\s*引\s*号\s*[：:]')

# 需要提取的字段名（按顺序）
TARGET_FIELDS = ['主诉', '现病史', '既往史', '评估', '体格检查', '辅助检查']


def is_field_start(line):
    """判断一行是否是某个字段的开头，返回字段名或 None"""
    stripped = line.strip()
    if not stripped:
        return None
    for field_name, pattern in ALL_FIELD_PATTERNS.items():
        if pattern.match(stripped):
            return field_name
    return None


def extract_age_gender(lines):
    """从文件内容中提取 年龄 和 性别"""
    for line in lines:
        # 查找包含 "年龄：" 的行
        match = re.search(r'年龄[：:]\s*(\d+岁)\s*性别[：:]\s*([男女])', line)
        if match:
            return f"年龄：{match.group(1)}性别：{match.group(2)}"
    return None


def extract_fields(filepath):
    """从txt文件中提取指定字段"""
    try:
        # 尝试多种编码读取
        content = None
        for encoding in ['utf-8', 'gbk', 'gb2312', 'gb18030', 'latin-1']:
            try:
                with open(filepath, 'r', encoding=encoding) as f:
                    content = f.read()
                break
            except (UnicodeDecodeError, UnicodeError):
                continue
        
        if content is None:
            print(f"  [错误] 无法读取文件: {filepath}")
            return None

        lines = content.split('\n')
        
        # 1. 提取年龄和性别
        age_gender = extract_age_gender(lines)
        
        # 2. 提取各字段内容
        # 策略：遍历每一行，判断属于哪个字段，收集目标字段的内容
        current_field = None
        field_contents = {}
        
        for line in lines:
            field_name = is_field_start(line)
            
            if field_name is not None:
                current_field = field_name
                if field_name in TARGET_FIELDS:
                    field_contents[field_name] = [line.rstrip()]
            else:
                # 当前行不是新字段的开始，属于上一个字段的延续
                if current_field in TARGET_FIELDS:
                    stripped = line.rstrip()
                    if stripped:  # 忽略空行
                        field_contents.setdefault(current_field, []).append(stripped)
        
        # 3. 组装结果
        result_lines = []
        if age_gender:
            result_lines.append(age_gender)
        
        for field in TARGET_FIELDS:
            if field in field_contents:
                result_lines.extend(field_contents[field])
        
        if not result_lines:
            print(f"  [警告] 未提取到任何内容: {filepath}")
            return None
        
        return '\n'.join(result_lines)
    
    except Exception as e:
        print(f"  [错误] 处理文件时出错 {filepath}: {e}")
        return None


def find_txt_files(base_dir):
    """递归查找所有 .txt 文件（排除 -mask.txt 文件）"""
    txt_files = []
    for root, dirs, files in os.walk(base_dir, followlinks=True):
        for fname in files:
            if fname.endswith('.txt') and not fname.endswith('-mask.txt'):
                txt_files.append(os.path.join(root, fname))
    return txt_files


def generate_mask_path(txt_path):
    """根据原始txt路径生成-mask.txt路径"""
    base, ext = os.path.splitext(txt_path)
    return base + '-mask' + ext


def main():
    txt_files = find_txt_files(BASE_DIR)
    print(f"找到 {len(txt_files)} 个 txt 文件")
    
    success_count = 0
    fail_count = 0
    
    for txt_file in sorted(txt_files):
        mask_path = generate_mask_path(txt_file)
        print(f"\n处理: {txt_file}")
        
        result = extract_fields(txt_file)
        if result:
            with open(mask_path, 'w', encoding='utf-8') as f:
                f.write(result + '\n')
            print(f"  -> 已生成: {mask_path}")
            success_count += 1
        else:
            fail_count += 1
    
    print(f"\n{'='*60}")
    print(f"处理完成！成功: {success_count}, 失败: {fail_count}, 总计: {len(txt_files)}")


if __name__ == '__main__':
    main()

