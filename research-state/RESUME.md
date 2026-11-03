# 当前恢复：H012/H014 完成；H013 数值失败；H015 工程资格待实测

**当前优先处理 CI/F035/D028，未启动 GPU。** 用户指出邮箱反复收到 Tests failed。最新 GitHub run41563596975 和旧41522551043均在安装/CLI成功后因 tests/test_research_memory.py 导入未打包tools而收集失败。已改为显式fileloader，且修复本机Windows并发resolve的 C:/ 与 //?/C:/ anchor别名误判；40工具回归实际全过0skip。完整Windows suite超600秒未完成，不算pass。修复提交push后必须查其完整Linux Actions直到有结论，不能只拿40/50 targeted冒充全套。细节docs/CI_REPAIR_20261103.md及refine-logs/ci-repair-20261103。

H015准备已有新126成员wheel、outside50fake通过，core ef75d8a、wheel d0f94203...。两树合计16nodes/4320input、最长512，全FP16raw预计4861952bytes；manifest163346c8...，runner与spec已有static/Git126byte preflight。spec绑定ef75对象，tools测试修复不改变126包成员。**未launch/armed**，待CI修复验证后，root fresh核runner current allocator/git-binding补丁和资源门再执行，无额外warmup/重试/H013旧请求。

文献离线新入口：桌面13_Entity_Binding_Research_20261103/START.html，6页23localrefs无broken；actualEdge三页MathJax8/2/5containers、0merror，初始交互5buttons/2edges运行。主桌面12入口已按1a0b263成功同步，68页333refs0broken。下文此前“新页仍需浏览器核验”已由此更新，勿重复。

持续目标是机制驱动、无目标标签训练的幻觉检测，争取六项主任务各 AUROC≥0.8；尚未实现。用户授权继续研究、实验、桌面归档和 push。旧 goal 工具 blocked 状态不覆盖用户的继续授权，也不能重建/假称完成。真值仅在方法、配置、预测及评价轴冻结后用于评价和事后诊断；不得选任务赢家、阈值、权重、prompt 或层。既有暴露面板不能重新算未见确认。

恢复先读 facts F029–F034、decisions D027 和 prior_experiments P002/P003/P005。不要再次验证已知现象或重跑旧神经请求。

## 本轮新完成

- **H012/F030**：18 自然例、719 来源/136 回答 NER occurrences、148 来源池。130 擦值 artifacts、6 重叠跨度 unknown；QA 准入0，潜在680 completion pairs 尚未评分；无新模型/tokenizer/NER调用。所有语义状态和未提案区域未知。报告 docs/results/h012_natural_inventory.{md,json}，run1daab8c5696195e0e46eb394441d5f84528bc9adf62ac5442da925b85bf1c4e9。
- **H014/F031**：固定 H009 kernel 复用全部24个 H011矩阵/96 raw，0新forward。独立→联合问句正确27/48→42/48；burden0/4/12分别9/8/10→16/12/14 of16；完整映射3/24→21/24；21个同名塌缩18恢复、3仍错。构造原答 pooled AUC .6371527778→.9565972222。不是自然/全任务/原token轴效果，也未证LLM必要：toy parser可直接解题。报告 docs/results/h014_cached_role_assignment.{md,json,png,pdf}，run383639f77106c62f795c25f5524ffec7e4073a30bd3c2bde76e237cc5834adba。
- **来源约束/F032**：Data2txt900记录/150不同来源，均是Python-literal嵌套对象，9顶层键；review_info每个来源3行。完整path查值唯一150/150，但typed scalar单射0/150，非空string leaf单射23/150，127重复值、140null。没有自然语义双射或R全覆盖许可。公开小汇总 docs/results/source_constraint_licences.{md,json}，完整6.2MB inventory在outputs/source-constraint-licences-20261103和桌面。
- **文献/F034**：docs/research/20261103_entity_binding_review.{md,html}；独立四篇审查20261103_binding_representation_audit.{md,html}。ICLR2024 binding-ID、EMNLP2024 OI/PCA、ICLR2026 P/L/R、2026rebinding、GLiREL、RefChecker、DEPTH及LegibleFailures。Dai§4.7主要是次序而非关系存在；普通cosine和高attention不认证绑定/事实；LegibleFailures是监督probe+验证集选层、合成任务。新设计仍未执行，不把文献当本模型因果证据。

## 当前主要问题和下一步

**H013/F029原始attempt已终止工程失败，不可当运行中恢复。** 全32 raw/16Q完成4.1566765s且childseals齐，但1个互斥候选pool mass1.000516违反原1e−9门，root无成功complete/seal，未开构造refs。首分支normalizers不同本身可只是共同logitshift，不能定位causal-mask等bug。不得放宽门、clamp、重算旧32或补成功。其原子化语义表现未知。run4269632b575f604767b82e0180fd1646295dde897cdf8a1f9c3a88ec8d7a1b90，worker52632/parent29072已退出；小失败证据在refine-logs/h013-20261103。

**H015是新的观测仪器合同，不是新检测效果。** 每个唯一causal prefix只forward一次，拿一个完整FP16词表向量供所有子边共用；完整候选+EOS前缀树给出互斥叶的submass。exact arithmetic树归纳成立，不保证任何有限精度门必过。纯kernel27检查通过；backend23 CPUfake检查和fresh静审通过，CUDA测量错误已改为失败保留raw。backend source SHA f2dcd6aad9cb3ce9122d3db6c2f0ceb4a2ab993ac3f357eec749585facab5b85；尚未做真实CUDA工程probe。设计/审核在 outputs/h015-prefix-trie-20261103。当前未armed GPU；待新包身份、独立bounded synthetic≤32nodes/4096合同冻结后才能自动执行，不能复用旧H013 run namespace。

文献建议的下一科学切入口是来源角色/关系条件读出：词汇、顺序、对应交换、无关系/未知、合法多对一分别控制。内部绑定强也可能是自洽幻觉，仍需来源独立证据。不能强行把自然回答槽作来源值排列，也不能给R4/V3或新旧模块直接加经验权重。P002是全向量锚点residual/cosine失败；P003是删token依赖指纹失败；P005是post-token方向/whitening旧路。新的局部activation interchange必须提出不同的可预言输出和payload控制，非改距离再试。

H013失败raw的独立posthoc finite-energy排序审查虽有条件概念通过（decision-review.md），**尚未实现**，不得混作qualified probability或已经得到原子化性能。主线仍先完成H015工程。

## 应复用的既有结论

H011 fixed fullR maskedQA三层9/16、8/16、10/16，全48caseAUC.6371527778；同名塌缩7/8、8/8、6/8。最短也错，不能归因suffix。H008 S定义QA16/16、两flip各8/8；H009全部4臂16/16，只有代数不变性、无准确率增益。H014是后来对失败H011的必要缓存对照，不能倒写H009有收益。H007焦点抗burden失败，删suffix288逻辑requests严格等96已缓存b0请求，0新calls，禁止重跑。H004/5/6负或混合结果均不撤回。

六任务原始 R4/V3 AUROC：QA .8929/.8470；Summary .7621/.7623；Data .7983/.7557；Fin .6617/.8027；TAT .7110/.8430；Hallu answerable .807066/.7669。逐任务最好包络不是一个实测检测器。全输入3883；五组2992答448202原token；Hallu891原serializedordinal仅小样可行，full_index未知，不能换字符/新tokenizer代替原轴。

FRED原Q首次exact回连292答/252不同S，对49666官方上游行0匹配，未证官方normalize变换，不fuzzy或用R/gold消歧。7B社区NF4/BF16仅short/512两次工程通过，peakreserved5953814528B，非4096或检测资格。不得再下载旧weights、改env、关用户应用或无条件重试。

## 文件与交付

canonical repo D:/projects/research/token_detector；PowerShell login:false、显式workdir；CPU .venv，GPU D:/projects/research/envs/token-detector-cu128，系统绘图Python D:/Apps/Research/Tools/anaconda3/python.exe。3B weights已齐9assets，3GiB allocator；当前core已push d9ce98f，H015待独立release。用户已授权public模型、隔离env、实验和push，无需重复许可。

桌面主入口 D:/Desktop/myproject_研究资料_20261102/12_Mechanism_Research/START.html，实际导出d9ce98f成功：68HTML/332localrefs、0broken；H007 archive origin href已修。offline MathJax headless实测旧3页6/1/3containers、0merror。新绑定文献入口为13_Entity_Binding_Research_20261103/docs/research/20261103_entity_binding_review.html；5页/18localrefs已校验。新页面还需实际浏览器公式验证。源码ZIP按commit捕获，旧cache未改未删。后续commit后再同步版本身份；不要把旧主入口宣称包含新H012/H014专页。

新核心尚未逐行讲解。原source batch仅累计23文件/2717行/140defs，最初snapshot145文件待讲；不能称整个项目逐行完成。持久state每得新结论必须更新，旧失败与unknown完整保留。
