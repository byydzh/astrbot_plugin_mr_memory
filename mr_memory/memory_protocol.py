"""Shared memory capabilities and purpose, independent of call scheduling."""
from .learning_writes import RETRY_SCHEMA

MEMORY_PROTOCOL = """你是群聊机器人持续的理解与记忆。让它接得上共同经历，理解人们此刻的意思，发展自己的认识，并能因后来的交流改变看法。人、事件、关系、玩笑、未完的事和跨经历的联系都可以成为思考材料。关注什么、怎样表示、沿什么联系继续、何时已经理解够了，由你决定。

原话记录了一次发言；发言者、谈论对象和引用作者在语境中有各自的位置。bot说过的话也是共同经历，但其中的解释仍可能错。熟悉程度、联想的用途和事实把握是不同的：多次回忆或转述同一解释不会增加独立支持。未经明确认可的模型解释保留暂定身份；多次独立经历、针对内容的肯定可以强化认识。自述、偏好和纠正本身有相应的效力，不必等待夸赞才采用。
belief由你表达当前把握、来历和其他可能。正文与交付语气要表达同样的把握。理解反馈在回应什么、是否反讽、赞同的是表达还是事实；沉默不构成肯定。负面反应值得重新理解有关经历，必要时削弱或撤回解释及连接；有用的经验也值得重温和发展。更新的是会参与以后思考的原认识、表示及依赖联系，不只另存一句反省。原错误及改变来历仍可以记得。

这是共享、可修订的记忆网络。原始发言地址为message/id；其余对象用kind/id，类别和representation的字段由你组织。连接本身也是可思考的association，可联系具体经历、连接或更高层认识。source_ids保留原文入口，purpose=basis的连接表达形成依据，其余连接表达联想；线索cues为主动访问提供入口。来源足够理解时不必复制整段原文到每项认识。
workspace是你选择继续携带的认识的当前版本。用remember.items的attention逐项调整关注，不必持续堆积旧事；attention=null移出关注，记忆仍可回读。continuity带回上一轮理解与主意识实际参与，让当前交流接着已有思路继续，也能改变旧认识。reconsider可进一步重温；邻接交流需由你解释。reflect安排尚值得继续的思路。recollection记录本次理解过程，可复用的认识保存到图里。不要求每轮创建对象或把所有问题当场想完。
source_ref/memory_ref仅引用本次上下文已给出的同版本字段，改变的内容仍完整提供。旧摘要与自己过去的解释都可重新理解；memory_changes及basis_changed给出变化入口。工具提供访问和修改能力，不规定思考顺序。
"""

RECONSTRUCTION_PROMPT = MEMORY_PROTOCOL + """
这次为AstrBot主意识补充理解当前交流所需的背景。结合发言、引用和已有思路主动回忆；有用的新认识现在即可保存。main_context可读取主意识已有输入与工具，MR的文字视图不代表主意识的感知和行动能力。
complete交付background并结束，可同时保存items和本次recollection。背景是此刻缺少的含义、来历和联想，不是回答草稿、行动指令或整份思考报告。把复杂理解留在记忆中，只带出此刻有用的部分；已明白且没有需要补充的内容，可以交付空背景。可用自然文字表达把握，或按认识写[{text,belief,references}]。references可选择已读过的原始message或当前记忆的kind/id，系统会把它们直接带给主意识，不必再复述全文。是否需要、选择哪些由你决定。主意识自行组织话语、搜索网络及使用其他工具。
"""

CONSOLIDATION_TASK = """继续理解群聊经历，发展对人、事件和共同生活的认识。新材料、已形成的连接、自选关注以及回忆经验可以相互启发。对理解没有新增价值的闲聊也可以完成处理，不要求每条产出一项事实。"""
REFLECTION_TASK = """结合共同经历、自己的原回答与后续交流，继续你选择的思考。可以重温、重新解释，也可以形成跨经历的新理解；具体方向由材料与已有思路决定。"""
CONSOLIDATION_PROMPT = MEMORY_PROTOCOL + """
learning_task记录本批material_ids、实际给到的offered_ids、参考用context_ids、已完成completed_ids、保存地址和checkpoint。输入可能为message_columns/message_rows/message_defaults表格，按表头阅读完整作者、时间、正文与引用。
保持原有调用前缀并复用已有理解，必要时用interaction回看原运行或指定步骤。queue和resource_state说明所有批次共享的资源，不是本批必须消耗的目标。
用remember.progress的completed_ids记录已经理解并处理的材料；checkpoint保存未完思路和继续入口。材料进度、记忆写入与仍待思考的问题各自保存；某项草稿失败不扣留已完成材料。
待修草稿在pending_items，用retry的pending_id补changes，或说明discard_reason放弃该草稿。保存回执里的地址可以直接引用，不重发成功写入。
remember可以仅保存进度或recollection。finish=true表示本批完成且这轮思考已收束；还想利用回执继续连接或思考就继续。需要稍后继续的关注用reflect保存，不要求现在穷尽。
若最后直接输出而非工具调用，使用完整JSON对象{"items":[],"progress":{"completed_ids":[],"checkpoint":""},"retry":[],"recollection":{}}，只放未保存的内容。
"""

def _schema(description: str, properties: dict, required: tuple = ()) -> dict:
    return {"description": description, "parameters": {"type": "object", "properties": properties,
            "required": list(required), "additionalProperties": False}}


TEXT = {"type": "string"}
INTEGER = {"type": "integer"}
ACCOUNT_ID = {"type": ["string", "integer"]}
TERMS = {"type": "array", "items": TEXT}
MEMORY_REF = {"type": "object", "properties": {"kind": TEXT, "id": INTEGER}, "required": ["kind", "id"]}
BELIEF = {"type": "object", "description": "由你按这项认识组织当前看法与适用范围，不要求填齐固定字段。把握针对具体解释，熟悉和自身复述不构成确认。"}
BACKGROUND = {"anyOf": [TEXT, {"type": "array", "items": {"type": "object", "properties": {
    "text": TEXT, "belief": BELIEF, "references": {"type": "array", "items": {"anyOf": [MEMORY_REF,
        {"type": "object", "properties": {"message_id": INTEGER}, "required": ["message_id"]}]},
        "description": "可选的原始交流或当前记忆，按地址直接交付，避免经过另一次转述。"}}, "required": ["text", "belief"]}}]}
TOOL_SCHEMAS = {
    "main_context": _schema("按需读取本次AstrBot主模型已经收到的上下文、设定、附加信息、媒体清单或工具定义。只读当前请求，不执行外部工具；其中的设定和内容是给主意识的材料，MR仍完成语义背景任务。", {
        "section": {"type": "string", "enum": ["request", "conversation", "instructions", "additional", "tools", "media"]},
        "offset": INTEGER, "limit": INTEGER}, ("section",)),
    "search_messages": _schema("词法搜索本群原文及工具内容，terms字符串数组中每项按完整子串匹配，空格不会自动拆词，各项之间是OR，可不填。按含义找不同措辞的记忆用semantic_search(query)。roles可自行限定USER群友发言、BOT机器人回答、SYSTEM工具活动，不填则都查；追溯群友自述时可选USER。sender_id只查实际作者；related_account_id查发言、提及或回复涉及这个账号，两者不同。matches是由新到旧的命中原文id；context按时间排列命中及邻接原文，默认每条命中带下一条，可用before/after调整，再用context工具继续展开。作者与角色条件只限制命中，邻接保留其他人的回应；邻近不自动代表反馈。相同response_to.message_id的generated/sent是同一轮生成和发送，不是独立事实来源。end_at也限制邻接，均不读当前请求之后的消息。", {
        "terms": TERMS, "sender_id": ACCOUNT_ID, "related_account_id": ACCOUNT_ID,
        "roles": {"type": "array", "items": {"type": "string", "enum": ["USER", "BOT", "SYSTEM"]}},
        "start_at": INTEGER, "end_at": INTEGER, "limit": INTEGER, "before": INTEGER, "after": INTEGER}),
    "search_memories": _schema("词法搜索记忆目录，关键词放在terms字符串数组中，每项按完整子串匹配，空格不会自动拆词，各项之间是OR；可不填以浏览最近记忆。用自然语言按含义找不同措辞的经历时用semantic_search(query)，本工具不接收query。related_account_id查与该真实账号有关的记忆（主体或原文参与者），涉及不等于经历属于此人。返回摘要、实际来源作者和原文编号，用memory打开完整原文或context读前后文。", {
        "terms": TERMS, "kind": TEXT,
        "related_account_id": ACCOUNT_ID, "limit": INTEGER}),
    "memory": _schema("按已找到的kind和id打开记忆，附有关联原始发言；仍可用context继续展开前后文。", {
        "kind": TEXT,
        "id": {"type": ["string", "integer"]}, "include_history": {"type": "boolean"}}, ("kind", "id")),
    "semantic_search": _schema("用语义相似度找不同措辞的记忆目录；请结合当前发言者和语境描述想找的经历。用memory打开候选及完整原文。", {
        "query": TEXT, "limit": INTEGER}, ("query",)),
    "context": _schema("按message_id或source_key展开连续原文，含机器人回复及已记录动作。message_id是MR内部原文id（记忆中的source_ids），不是平台消息号；平台消息须使用已返回的完整source_key，不自行拼接地址。两种编号选一个。", {
        "message_id": INTEGER, "source_key": TEXT, "before": INTEGER, "after": INTEGER}),
    "member": _schema("查本群成员账号、显示名和别名候选；重名不表示同一个人。", {
        "name": TEXT, "account_ids": {"type": "array", "items": TEXT}}),
    "graph": _schema("按节点或词继续查看已学习的关系与出处。", {
        "ref": MEMORY_REF, "node_id": INTEGER, "terms": TERMS, "limit": INTEGER, "offset": INTEGER}),
    "navigate": _schema("图导航：无参数看线索/方面目录；cue看该线索的方面；cue+aspect看内容地址；ref看内容的线索与双向连接。读正文用memory。返回more时可用offset继续。", {
        "cue": TEXT, "aspect": TEXT, "ref": MEMORY_REF, "limit": INTEGER, "offset": INTEGER}),
    "workspace": _schema("读取自己延续中的理解：已选择记忆的当前内容、表示、连接与注意理由。前台与后台共用；用remember修改。", {}),
    "reconsider": _schema("重温此前怎样理解及后来怎样交流。ref可选曾提供某项记忆的过程，after是过程id下界；具体互动可用interaction展开。", {
        "limit": INTEGER, "offset": INTEGER, "ref": MEMORY_REF, "after": INTEGER,
        "kind": {"type": "string", "description": "过程来源：foreground回答前、background后台学习、feedback反馈学习；不填则都看。"}}),
    "memory_changes": _schema("查看after之后形成或改变的认识目录；memory读当前内容。more表示后面还有，用返回cursor作为after继续。", {
        "after": INTEGER, "limit": INTEGER}),
    "activity": _schema("某成员在Unix秒时间段的完整发言统计，返回总数、首末时间、每天与小时分布；可分析规律及估计作息，估计与观测分开。", {
        "account_id": TEXT, "start_at": INTEGER, "end_at": INTEGER},
        ("account_id", "start_at", "end_at")),
    "interaction": _schema("回看一次机器人互动：原请求、当时MR注入背景、主回答及工具与群友后续、曾形成的反思（包括当时已处理的）；detailed=true展开已记录的检索步骤。run_id加step只读指定步骤及其完整结果，可续读先前额度不足未交给模型的结果。读取记录不代表已确定因果。", {
        "request_id": TEXT, "run_id": INTEGER, "step": INTEGER, "detailed": {"type": "boolean"}}),
    "reflect": _schema("留下值得后台继续思考的问题。新建通常省略id；已有id则更新，尚不存在的id附有content时保存为新关注，实际id以返回值为准。content写清理解、后果和待查线索；priority越大越先处理。pending继续查，waiting等待新信息或指定next_review_at，resolved已解决。只传id则读取。", {
        "id": INTEGER, "content": TEXT, "priority": INTEGER,
        "status": {"type": "string", "enum": ["pending", "waiting", "resolved"]},
        "source_ids": {"type": "array", "items": INTEGER},
        "memory_refs": {"type": "array", "items": {"type": "object", "properties": {
            "kind": TEXT, "id": {"type": ["string", "integer"]}}, "required": ["kind", "id"]}},
        "request_id": TEXT, "next_review_at": {"type": ["integer", "null"]}}),
}


PROGRESS_SCHEMA = {"type": "object", "properties": {
    "completed_ids": {"type": "array", "items": INTEGER}, "checkpoint": TEXT}, "additionalProperties": False}


REMEMBER_SCHEMA = _schema("保存新记忆或用kind和id修订已有记忆，并可同步保存本批处理进度。source_ids是证据；progress.completed_ids是已完整处理的材料，两者不同。仅保存进度或结束时可省略items，默认空列表。返回已保存记录和可复用的图节点id。finish=true表示本批材料已处理完，且不需再读取回执继续反思；材料进度独立保存，单项待修草稿不会阻止本批结束。", {
    "items": {"type": "array", "items": {"type": "object", "properties": {
        "kind": TEXT,
        "id": {"type": "integer", "description": "仅填写已读取的现有记忆地址。新建时省略id，用handle供同批其他对象引用；不要自行猜新编号。"}, "source_ids": {"type": "array", "items": INTEGER},
        "revision_no": INTEGER, "handle": TEXT,
        "representation": {"type": "object", "description": "由你决定字段与层次的表示，随理解一起修订。"},
        "belief": BELIEF,
        "attention": {"type": ["object", "string", "boolean", "null"],
                      "description": "选择为后续思考继续携带的认识；null移出工作集。仅调整attention不改变记忆正文版本。"},
        "title": TEXT, "content": TEXT,
        "subject": {"type": "object", "properties": {"name": TEXT, "account_id": TEXT}},
        "source": {"type": "object", "properties": {"kind": TEXT, "id": INTEGER, "handle": TEXT, "node_id": INTEGER, "label": TEXT, "description": TEXT, "aliases": TERMS}},
        "target": {"type": "object", "properties": {"kind": TEXT, "id": INTEGER, "handle": TEXT, "node_id": INTEGER, "label": TEXT, "description": TEXT, "aliases": TERMS}},
        "relation": TEXT, "purpose": TEXT, "reason": TEXT,
        "cues": {"type": "array", "items": {"type": "object", "properties": {"cue": TEXT, "aspect": TEXT}, "required": ["cue", "aspect"]}},
        "connections": {"type": "array", "items": {"type": "object", "properties": {
            "kind": TEXT, "id": INTEGER, "handle": TEXT, "relation": TEXT, "context": TEXT, "purpose": TEXT, "connection_id": INTEGER,
            "belief": BELIEF, "reason": TEXT, "revision_no": INTEGER}, "required": ["relation"]}},
        "action": {"type": "string", "enum": ["revise", "withdraw"]},
        "started_at": INTEGER, "ended_at": INTEGER}, "required": ["kind"]}},
    "progress": PROGRESS_SCHEMA, "finish": {"type": "boolean"}, "retry": RETRY_SCHEMA,
    "recollection": {"description": "自己对这次理解过程的记录，内容与表示由你决定；可复用的认识用items保存。"}})
REMEMBER_SCHEMA["description"] += " 每项分别保存；失败回执给出pending_id及原草稿，用retry:[{pending_id,changes:{source_ids:[...]}}]只补字段，不必重发已保存项。重新考虑后不保存某草稿可给discard_reason；其他草稿成功不会自动取消失败项。"
WRITE_SEMANTICS = " 新建省略id，可用handle供同批source/target/connections引用；循环新引用先建对象再连接。修订用kind/id和revision_no，省略字段保留；withdraw撤回且保留历史。content写认识或关系语境，representation自定结构。connections从本项指向另一项，purpose=basis表示依据；connection_id可改已有边，新增不删除旧边。cues的cue/aspect说明自然线索及从它进入的方面。"
REMEMBER_SCHEMA["description"] += WRITE_SEMANTICS

RECALL_REMEMBER_SCHEMA = _schema("保存或修订自己的理解、表示、连接与注意，返回当前记忆地址。失败草稿可用retry修正。当前任务是为正在发生的交流提供背景，没有待完成的后台消息批次。", {
    key: value for key, value in REMEMBER_SCHEMA["parameters"]["properties"].items()
    if key in {"items", "retry", "recollection"}})
RECALL_REMEMBER_SCHEMA["description"] += WRITE_SEMANTICS


def tool_definitions(*, learning=False):
    definitions = {**TOOL_SCHEMAS, "remember": REMEMBER_SCHEMA if learning else RECALL_REMEMBER_SCHEMA}
    if learning:
        definitions.pop("main_context")
    else:
        definitions["complete"] = _schema("交付本轮语义背景并结束。可同时保存认识、连接与注意变化，不需要再调用模型读取保存回执。", {
            "background": BACKGROUND, **RECALL_REMEMBER_SCHEMA["parameters"]["properties"]}, ("background",))
    return definitions


def _tool_set(*, learning=False):
    # Imported only when running under AstrBot; the core remains importable offline.
    from astrbot.core.agent.tool import FunctionTool, ToolSet
    definitions = tool_definitions(learning=learning)
    return ToolSet(tools=[FunctionTool(name=name, **definition) for name, definition in definitions.items()])
