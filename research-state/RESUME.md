# 当前恢复：H016 已完成；已写远端数据交接，等待本次 push/CI

**用户最新指示：写MD列出远端需要的数据位置并push，让另一台电脑继续。** 已写REMOTE_EXPERIMENT_HANDOFF.md，README有入口；路径来自历史plan/manifest/下载收据而非本次云端在线探测。prepared/native/score/model清单、原样迁移约束与本地近期制品分列。H006/H015/H016云端目录未确认，勿把D:位置当云端。

**H016/F038/D031已完成，不得重复。** Registry a52676148c96453c234c569ec0abc403a09ef9c3180113d7d05e6427a6683f70；controller38472/redirector38792及GPUchild已退出，launch-state armed=false。312actualforwards/86930input/0generated；maxcontext308；32.9832105s forward/capture，runner196.3688148s，peakreserved3072327680B。原obsseal caab060faed092b9bfc0378bb9364c02eb415e8ff9b6b032f64778f9a8277982；CPUevalseal ce5999b2cbc499473f77d309df6db62e93099dd089cd938e80f82e6554e743ea。全部4臂先freeze后读refs，CPU重放完整向量通过。

**固定四臂结果：** compoundind/joint、atomicind/joint角色正确9/16、16/16、15/16、16/16；完整来源正确1/8、8/8、7/8、8/8；16构造case次级AUROC .625、1、1、1。atomicind仍有一问错误，AUC1不代表全对。仅8source已暴露toy、parser可直接解答；不推广自然双射、不证明LLM必要/内部唯一因果/新颖性，不把四臂挑最好当六任务方法。docs/results/h016_paired_scope.{md,json,html}及桌面15_H016_Paired_Scope_20261103/START.html已交付；桌面含181sealed文件及全部32原NPZ，约105MB。

**代码/验证。** 核心0cc3c40，immutable wheel c2ede5...共129成员/126py+3resources；70项外部安装CPUfake checks通过。freshcore和deltareview完成，原硬件gate缺口已修复。全CPU套件在仓库外于180秒明确handoff上限停止，记录incomplete，不能称全套pass；要核对随后push的精确HEAD完整LinuxCI。此前1b7bf83的972pass只属于旧版本。refine-logs/h016-20261103保留byteexact准备、完整CPU评价、小型原obsmetadata；原32NPZ仍在outputs/桌面，不在Git。

**资源。** 用户允许正常关闭WPS，简历主窗口已退出，未强杀或丢弃文档。Edge已关，Termius/Quark未运行，主VSCode/Codex和系统/网络服务保留；无需继续关程序。所有H015/H016控制器结束，无armedGPU任务。H015/F037两输入16forward资格保留，不重跑；旧H013概率门失败不补成功。

下一步以用户换机交接为当前任务：完成batchpush、核对精确HEAD CI，并让新电脑先读取AGENTS/BRIEF/RESUME与研究memory，不重复已做toy。新自然方法需先冻结可核relation/predicate前端、合法重复/未知/作用域控制和完整人口，真值仅冻结后评价。目标六任务各AUROC≥.8仍未完成；旧goal工具blocked不覆盖继续授权，也不重建/假称complete。

## 本轮新完成

- **H012/F030**：18 自然例、719 来源/136 回答 NER occurrences、148 来源池。130 擦值 artifacts、6 重叠跨度 unknown；QA 准入0，潜在680 completion pairs 尚未评分；无新模型/tokenizer/NER调用。所有语义状态和未提案区域未知。报告 docs/results/h012_natural_inventory.{md,json}，run1daab8c5696195e0e46eb394441d5f84528bc9adf62ac5442da925b85bf1c4e9。
- **H014/F031**：固定 H009 kernel 复用全部24个 H011矩阵/96 raw，0新forward。独立→联合问句正确27/48→42/48；burden0/4/12分别9/8/10→16/12/14 of16；完整映射3/24→21/24；21个同名塌缩18恢复、3仍错。构造原答 pooled AUC .6371527778→.9565972222。不是自然/全任务/原token轴效果，也未证LLM必要：toy parser可直接解题。报告 docs/results/h014_cached_role_assignment.{md,json,png,pdf}，run383639f77106c62f795c25f5524ffec7e4073a30bd3c2bde76e237cc5834adba。
- **来源约束/F032**：Data2txt900记录/150不同来源，均是Python-literal嵌套对象，9顶层键；review_info每个来源3行。完整path查值唯一150/150，但typed scalar单射0/150，非空string leaf单射23/150，127重复值、140null。没有自然语义双射或R全覆盖许可。公开小汇总 docs/results/source_constraint_licences.{md,json}，完整6.2MB inventory在outputs/source-constraint-licences-20261103和桌面。
- **文献/F034**：docs/research/20261103_entity_binding_review.{md,html}；独立四篇审查20261103_binding_representation_audit.{md,html}。ICLR2024 binding-ID、EMNLP2024 OI/PCA、ICLR2026 P/L/R、2026rebinding、GLiREL、RefChecker、DEPTH及LegibleFailures。Dai§4.7主要是次序而非关系存在；普通cosine和高attention不认证绑定/事实；LegibleFailures是监督probe+验证集选层、合成任务。新设计仍未执行，不把文献当本模型因果证据。

## 当前主要问题和下一步

**H013/F029原始attempt已终止工程失败，不可当运行中恢复。** 全32 raw/16Q完成4.1566765s且childseals齐，但1个互斥候选pool mass1.000516违反原1e−9门，root无成功complete/seal，未开构造refs。首分支normalizers不同本身可只是共同logitshift，不能定位causal-mask等bug。不得放宽门、clamp、重算旧32或补成功。其原子化语义表现未知。run4269632b575f604767b82e0180fd1646295dde897cdf8a1f9c3a88ec8d7a1b90，worker52632/parent29072已退出；小失败证据在refine-logs/h013-20261103。

**H015是新的观测仪器合同，不是新检测效果。** 每个唯一causal prefix只forward一次，拿一个完整FP16词表向量供所有子边共用；完整候选+EOS前缀树给出互斥叶的submass。exact arithmetic树归纳成立，不保证任何有限精度门必过。纯kernel27检查通过；backend23 CPUfake检查和fresh静审通过，CUDA测量错误已改为失败保留raw。backend source SHA f2dcd6aad9cb3ce9122d3db6c2f0ceb4a2ab993ac3f357eec749585facab5b85；两synthetic输入真实CUDA工程probe已完成，见本页顶部F037。设计/审核在 outputs/h015-prefix-trie-20261103。H015 worker已完成并退出，勿重新launch；当前H016未运行，不能复用旧H013 run namespace。

文献建议的下一科学切入口是来源角色/关系条件读出：词汇、顺序、对应交换、无关系/未知、合法多对一分别控制。内部绑定强也可能是自洽幻觉，仍需来源独立证据。不能强行把自然回答槽作来源值排列，也不能给R4/V3或新旧模块直接加经验权重。P002是全向量锚点residual/cosine失败；P003是删token依赖指纹失败；P005是post-token方向/whitening旧路。新的局部activation interchange必须提出不同的可预言输出和payload控制，非改距离再试。

H013失败raw的独立posthoc finite-energy排序审查虽有条件概念通过（decision-review.md），**尚未实现**，不得混作qualified probability或已经得到原子化性能。主线仍先完成H015工程。

## 应复用的既有结论

H011 fixed fullR maskedQA三层9/16、8/16、10/16，全48caseAUC.6371527778；同名塌缩7/8、8/8、6/8。最短也错，不能归因suffix。H008 S定义QA16/16、两flip各8/8；H009全部4臂16/16，只有代数不变性、无准确率增益。H014是后来对失败H011的必要缓存对照，不能倒写H009有收益。H007焦点抗burden失败，删suffix288逻辑requests严格等96已缓存b0请求，0新calls，禁止重跑。H004/5/6负或混合结果均不撤回。

六任务原始 R4/V3 AUROC：QA .8929/.8470；Summary .7621/.7623；Data .7983/.7557；Fin .6617/.8027；TAT .7110/.8430；Hallu answerable .807066/.7669。逐任务最好包络不是一个实测检测器。全输入3883；五组2992答448202原token；Hallu891原serializedordinal仅小样可行，full_index未知，不能换字符/新tokenizer代替原轴。

FRED原Q首次exact回连292答/252不同S，对49666官方上游行0匹配，未证官方normalize变换，不fuzzy或用R/gold消歧。7B社区NF4/BF16仅short/512两次工程通过，peakreserved5953814528B，非4096或检测资格。不得再下载旧weights、改env、关用户应用或无条件重试。

## 文件与交付

canonical repo D:/projects/research/token_detector；PowerShell login:false、显式workdir；CPU .venv，GPU D:/projects/research/envs/token-detector-cu128，系统绘图Python D:/Apps/Research/Tools/anaconda3/python.exe。3B weights已齐9assets，3GiB allocator；H015核心ef75d8a及CI修复1b7bf83均已push；126成员新wheel已通过outside50fake。用户已授权public模型、隔离env、实验和push，无需重复许可。

桌面主入口 D:/Desktop/myproject_研究资料_20261102/12_Mechanism_Research/START.html，实际导出1b7bf83成功：68HTML/333localrefs、0broken；H007 archive origin href已修。offline MathJax headless实测旧3页6/1/3containers、0merror。新绑定文献入口13_Entity_Binding_Research_20261103/START.html：6页/23localrefs已校验；actualEdge三报告MathJax8/2/5containers、0merror，初始交互5buttons/2edges。源码ZIP按commit捕获，旧cache未改未删。后续commit后再同步版本身份；不要把旧主入口宣称包含新H012/H014专页。

新核心尚未逐行讲解。原source batch仅累计23文件/2717行/140defs，最初snapshot145文件待讲；不能称整个项目逐行完成。持久state每得新结论必须更新，旧失败与unknown完整保留。
