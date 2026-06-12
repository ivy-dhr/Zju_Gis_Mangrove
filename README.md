非常感谢Mr.Chris，帮我找回了失踪的RawData文件，以及Mr.Chris对我本次作业的指导
作为小白，完成本次2010-2025年的时序分析很有成就感
简要叙述项目逻辑：
arc.py用来处理，从https://zenodo.org/records/6894273下载的红树林矢量文件，得到待处理网格区域的对应矢量文件
javascriptForGEE.txt是用于  GEE平台训练模型，推测提取2025年红树林面文件（输出tif）文件
最后用tifpointsfist.py文件运行得到目标预测的红树林
data2025pre_processing是栅格文件转面文件
