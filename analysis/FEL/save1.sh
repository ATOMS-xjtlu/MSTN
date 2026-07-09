#!/bin/bash
# 保留每行第二个值及其前的空格，删除其他所有内容
# 用法：./process_file.sh 输入文件 [输出文件]（不指定输出文件则覆盖原文件）

input_file="$1"
output_file="${2:-$input_file}"  # 若未指定输出文件，则覆盖输入文件

# 创建临时文件确保安全处理
temp_file=$(mktemp) || exit 1

# 核心处理：使用正则匹配保留第二个值及其前的空格
perl -ne '
    if ($_ =~ /^\s*?\S+(\s+\S+)/) {  # 匹配第二个值及其前的空格
        print "$1\n";                 # 成功匹配时打印捕获组
    } else {
        print "\n";                   # 匹配失败时保留空行
    }
' "$input_file" > "$temp_file"

# 移动临时文件到目标位置
mv "$temp_file" "$output_file" && \
echo "处理完成！输出文件: $output_file"
