任务：

类似树叶处理一样，做二值化处理。

1. 6张图片分别处理，各出2个结果

    1张，是二值化的图
    1张，是道路保留，其它都去掉的彩图。

2. 提取出道路部分，仿照canny.jpg中 红线所画的 A 和 B部分。

   其余部分不要。
  
   其实就是将道路保留，道路中间的植物，道路两侧的植物一律去掉。

3. 出来的图，是二值化的图。

4. 前方，最高处，以天际交界之处为准。