# 当前恢复：M3负结果、M4自然观测与旧读出链审计完成；新方法未准入

主目标仍为已有回答的无监督幻觉检测、六任务各AUROC≥0.8；真值只在评分冻结后评价。当前没有新risk/AUROC，也没有达到目标。先读F050/F051与docs/results/h017_relation_binding.md；读出诊断已知，不再换名重做。

**无活动模型/autorun job。** M4 run31185253c6389623bea031a07db70d77043ee67315e9fe85ca426725dc826d32已完成、seal、finish supported_scope（仅合格描述性观测）。完整441-token原Q/S/R，15716/source15540，1次全32层native+1次76行head，67.235607s，77个原query全部保留。原worker35556/parent52928已退出；M4_FINAL_PROGRESS.json为最终状态，不得按旧LIVE_PROGRESS重启。

**自然结果F050：** 源over300yearsBC→Raround300yearsago。p300=.997988、years=.925592，ago仅.000114724/rank5；同一preago位置首选BC，p=.982174，logit领先9.05499649。preago全部1024layer/head平均BC mass .054506、300 .0041493，只有184/1024的300>BC。不支持“错误单位高置信且只看数字漏掉BC”。M4无因果干预，不能据此归因原Mistral生成器、条件语义确切编码或做新检测器。posttoken轨迹已观测，source-reading Q未观测。

原15层全部坐标与全部32层AV资格/原轴通过；FP32 logp/p元数据逐位复现。严格独立FP64经验界6/76未过（maxlogp6.045e-5/p2.518e-5/H1.157e-4），仍inconclusive，失败原样保留且未放宽；三词高低判断在两精度一致。源结果m4-natural-observation，独立m4-interpretation-supplement-v1，解释M4_POSTRUN_INTERPRETATION_REVIEW.md。

**旧链F051是另一个合同：** BF16源pago=.000121/无源=.003245，fullgap+3.28895排名1165、local+2.86277排名3975、actualpoint+3.07586排名1492。R4在22-token整句prior先降至116867，最终114453；19/22差值负。V3自身证据2.632417 + context−2.283023 = .349394，排名20219，超过top5%预算6241。旧单位告警本来存在，不能继续假设必须发明更强BC检索。原365prompt/76R IDs与M4相同，但BF16/FP32 logp固定.002比较44/76失败，停止跨运行因果桥；不可把M4内部trace当成旧分数的同次状态。primitive与native delta两元素差≤4.44e−16严格等式未过，原样保留；native delta与实际fullgap逐位一致。只引用旧链，不重新评分/排名。

**下一问题：** 如何用新的关系特异证据保留局部条件矛盾，又不误报合法改写和运算？不能直接裁负context、取max、加权救15716。需独立审查语义受控来源对照/共同内容抵消等候选，完整tuple许可/保义控制/原轴risk+unknown先定义；当前未准入新算法或模型job。M4只一条已暴露案例，不是新确认集；人工source-only25tuple不是自动前端。Pollux父亲未知；flag对象/路径/先后不补齐，不强配两个主体数字。

**已完成不重跑：** F044/F047的M1重放与M2对齐真实3+897=900QA/150来源；五错误实现各900拒绝，语义900unknown。M0=2700答/450来源、5806显式Data2txt字段。F045原252prompt两世界prefix2calls、F046fullhead/cache/noop4calls/3head均通过。F048的48model/44head受控K-role主门4/4失败、V-value4/4方向正，但正常toy没有高p错误，不选新头/层/prompt救援。累计H01755originalmodels=2earlystop+53full、48head、0目标标注/新risk。

最新代码不可变137target D:/projects/research/outputs/h017-qualified-20261103-191208/target，wheel fe8d9103429addc19fa25291b781bbb1b0d73ab7569e6833a7efb9c32af6c2ef。109relatedCPU通过；本机全套136达到900s、137达到180s均incomplete，不能说全套通过。最终提交必须看完整LinuxCI确切SHA并修复失败，旧525fd28的992通过不能替代。图/公式离线docs/source-reading/H017_PROGRESS.html，数学现15处排版；科学/审计负或inconclusive不等于软件CI失败。

模型9资产16.07GB与31GB所选数据均已独立验证，根C:/ResearchData/token-detector-remote-20261103（模型DATA04_models），不重复下载/搬动。M4原BF16存储/FP32算术，与旧mainline BF16计算要区分；源cache旧Q/K为preRoPE、只归档15且原执行0..15、R有T+1Q与全输入KV。新M4是R全32层Q/AV和全输入32层KV，仍不是全源Q/所有中间张量。仍以冻结R为对象，不回角色QA。
