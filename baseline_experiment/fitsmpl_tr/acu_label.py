import json

# 读取 JSON 文件
with open("/home/svt/HSMR/baseline_experiment/fitsmpl_tr/acupoint_definitions.json", "r", encoding="utf-8") as file:
    data = json.load(file)

# 将键名替换为编号
new_data = {str(index): value for index, value in enumerate(data.values())}

# 保存到新的 JSON 文件
with open("/home/svt/HSMR/baseline_experiment/fitsmpl_tr/acupoint_def_label.json", "w", encoding="utf-8") as file:
    json.dump(new_data, file, ensure_ascii=False, indent=2)

print("转换完成，结果已保存到 output.json 文件中。")