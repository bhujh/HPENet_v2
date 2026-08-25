# ball_query vs GridBallQuery 差别 + 大ball扫描冗余度

## 结论
1. 差别: 见对话回复对比表(源码 + plugin.md §15)
2. 大 ball 冗余度: 早 break 剪枝极有效,无大量冗余(plugin.md §15.4 近线性实测);
   真正可省的是跨-kernel 冗余(bq_dp 重算 dx/dy/dz),即融合1。
