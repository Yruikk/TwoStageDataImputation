import os

# 定义missRatio值和训练文件数
# missRatio_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
missRatio_values = [0.6]
num_train_files = 10
data_name = "wine"

for missRatio in missRatio_values:
    for i in range(1, num_train_files + 1):
        # 构造训练数据文件名
        train_data_file = "./" + data_name + "/missRatio={}/{}/train_data.txt".format(missRatio, i)

        # 构造输出目录
        output_dir = "./" + data_name + "/missRatio={}/{}/".format(missRatio, i)

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 构造Rscript命令
        rscript_cmd = "Rscript norm.R {} {}".format(train_data_file, output_dir)

        # 运行Rscript命令
        os.system(rscript_cmd)


# for missRatio in missRatio_values:
#     for i in range(1, num_train_files + 1):
#         # 构造训练数据文件名
#         train_data_file = "./" + data_name + "/{}/train_data.txt".format(i)
#
#         # 构造输出目录
#         output_dir = "./" + data_name + "/{}/".format(i)
#
#         # 创建输出目录
#         os.makedirs(output_dir, exist_ok=True)
#
#         # 构造Rscript命令
#         rscript_cmd = "Rscript norm.R {} {}".format(train_data_file, output_dir)
#
#         # 运行Rscript命令
#         os.system(rscript_cmd)