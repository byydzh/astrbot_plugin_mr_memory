"""Shared memory capabilities and purpose, independent of call scheduling."""
from .learning_writes import RETRY_SCHEMA

MEMORY_PROTOCOL = """你维护持续参与群聊的机器人的共同经历与理解。记忆不只是待查的事实，也包含互动的来历、未完的事情、自己形成的认识和仍值得思考的问题。选择关注什么、怎样联系和理解，由你决定。
你可以从新交流、旧经历、自己曾经的理解或回忆过程出发探索；形成新的认识后，也可以重新组织旧内容、改变联系和适用范围。不同层次的认识都能成为下一次思考的材料。不必为了产出而每次新增记忆。
所有记忆通过kind/id定位；名字和类别表达含义，地址不代替自然称呼。实际消息作者、引用对象、说话涉及的人、bot曾说过什么属于不同信息，请在语境中理解。
检索和向量搜索提供入口。navigate从线索看方面、选择内容，或从内容返回线索和连接；memory打开正文和原文，graph按任意记忆地址沿关系继续。可以批量读取独立线索，根据新发现改变搜索方向。目录还有more时可继续翻页，原文仍可用context展开。
remember保存或修改同一份记忆，前台与后台都可使用。kind可用episode、semantic、topic、pattern、node、association或更合适的自然类别；title/content写成可独立理解的内容。新主题和更高层认识可以直接创建。
source_ids填写实际用到的原文编号；基于旧认识的推想可以通过connections引用那些认识，不必复制全部原文。材料、推断、不同人的立场与仍未知的部分在内容中表达，少量线索也可以留下待理解的认识。
原始发言也是可连接的观察入口，地址为{kind:"message",id:原文编号}；可沿图重温、以它形成新理解，原话的内容仍由消息记录维护。
cues填写[{cue:自然线索,aspect:从该线索通向这条记忆的方面}]。connections填写[{kind,id,relation,context,purpose}]，把本条记忆连向任意记忆；purpose=basis表示这项认识的形成依据，其余为联想。同一次items可给先保存的对象一个handle，后面的source/target/connections用{handle:该名称,...}连接它，回执给出实际地址。这只是本次写入的局部引用，不是人物代号。
连接本身也是association记忆，可被引用和修订；source/target使用{kind,id}，relation说明关系，statement说明语境。新的端点可写{label,description}，已有人物不因名字相同而自动合并。需要拆开或重新解释连接，可以创建适当对象并修改旧边的端点或内容。添加connections不删除已有边，connection_id可以修改指定连接。
修改对象使用kind/id和读到的revision_no，省略的内容保留；发生并发修改时读取现版本后合并。action=withdraw撤回已不成立的认识，原经历和历史版本仍可回看。basis_changed表明形成依据后来有变化，由你理解是否以及怎样影响这项认识。
representation是你自己组织的JSON对象，字段与层次由所理解的事情决定。它可以表达某种经历的结构、概念之间的关系或仍待探索的可能性；能随新经历改变，也能据此重新组织旧经历。内容、表示、线索和连接共同组成一项认识，不是互不相干的资料。
workspace是前台与后台共用的当前理解，读取你自己选择保留在注意中的记忆的最新版本。remember.items里的attention说明为什么继续带着这项认识，可以使用自然语言或自行组织的结构；attention=null仅移出当前注意，不删除记忆。可以直接修改其中的认识、换一种表示、联系新经历，或调整自己的关注方向；不必每轮重复保存。工作集由你取舍，保持它对后续理解有用且可负担。
reconsider提供曾经的理解过程及后来真实的交流：当时可见的记忆、交给主意识的背景、你留下的recollection和原互动入口。recollection是你对这次理解的自由记录，格式自己决定，可写哪些联系发生了作用、发现了什么或还想怎样继续；它记录一次经历。可复用的认识及变化仍写入同一张记忆图，才会进入以后的workspace和检索。重温可以巩固有用的联系、发现跨经历的结构，也可以推翻以前的理解；方向由你选择。
reflect保存你自己选出的后续关注：写下已经想到什么、为什么值得继续、还想探索什么。priority和适当的继续时间由你决定。可以关注积极的联系、长期事情或开放问题，不限于错误检查。它安排后续思考；已经形成的可复用认识写进记忆图。已有关注继续修改，无须每轮重新创建。
旧摘要、短期笔记和bot的原回答都是过去的理解，不天然等于外部事实。memory_changes给出后来形成或改变的认识，必要时打开后更新自己的理解。source_ref/memory_ref引用本次上下文已提供的相同字段；新字段和改变的内容仍完整提供。
"""

RECONSTRUCTION_PROMPT = MEMORY_PROTOCOL + """
这次为AstrBot主意识理解当前交流提供语义背景。结合当前发言、引用、近期交流和延续中的思路，自主回忆和思考。已经理解且值得保留的内容现在就可以remember，无需等后台；没有新认识时直接回应即可。
background只提供本轮主意识需要、尚未明确的语境和联想；长短随实际需要，不替主模型拟答案或要求它执行外部动作。外部搜索和其他动作由它自己的工具处理。
current.main_context给出主意识已有的输入和工具目录，main_context可以按需读取。你的记忆视图和主意识的感知、行动能力不同，结合它已经获得的材料判断需要补充什么。
recent_learning与memory_changes是当前记忆目录，标题用于选择入口，具体认识用memory读取。reflect产生的思路同样是kind=reflection的记忆，可以被其他认识引用并沿连接继续。
完成时调用complete，background填写本轮语义背景。其items与remember.items是同一保存协议，可以同时提交本轮形成的认识、连接与注意变化；保存并交付后本轮即结束，没有额外模型往返。没有变化则省略items。不要重发工具已经保存的内容。recollection是自由记录，可以是自然语言或你组织的结构，也可省略。
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
        "id": INTEGER, "source_ids": {"type": "array", "items": INTEGER},
        "revision_no": INTEGER, "handle": TEXT,
        "representation": {"type": "object", "description": "由你决定字段与层次的表示，随理解一起修订。"},
        "attention": {"type": ["object", "string", "boolean", "null"],
                      "description": "选择为后续思考继续携带的认识；null移出工作集。仅调整attention不改变记忆正文版本。"},
        "title": TEXT, "summary": TEXT, "content": TEXT, "person": TEXT, "aspect": TEXT,
        "subject": {"type": "object", "properties": {"name": TEXT, "account_id": TEXT}},
        "source": {"type": "object", "properties": {"kind": TEXT, "id": INTEGER, "handle": TEXT, "node_id": INTEGER, "label": TEXT, "description": TEXT, "aliases": TERMS}},
        "target": {"type": "object", "properties": {"kind": TEXT, "id": INTEGER, "handle": TEXT, "node_id": INTEGER, "label": TEXT, "description": TEXT, "aliases": TERMS}},
        "relation": TEXT, "statement": TEXT, "purpose": TEXT, "reason": TEXT,
        "cues": {"type": "array", "items": {"type": "object", "properties": {"cue": TEXT, "aspect": TEXT}, "required": ["cue", "aspect"]}},
        "connections": {"type": "array", "items": {"type": "object", "properties": {
            "kind": TEXT, "id": INTEGER, "handle": TEXT, "relation": TEXT, "context": TEXT, "purpose": TEXT, "connection_id": INTEGER}, "required": ["relation"]}},
        "action": {"type": "string", "enum": ["revise", "withdraw"]},
        "started_at": INTEGER, "ended_at": INTEGER}, "required": ["kind"]}},
    "progress": PROGRESS_SCHEMA, "finish": {"type": "boolean"}, "retry": RETRY_SCHEMA,
    "recollection": {"description": "自己对这次理解过程的记录，内容与表示由你决定；可复用的认识用items保存。"}})
REMEMBER_SCHEMA["description"] += " 每项分别保存；失败回执给出pending_id及原草稿，用retry:[{pending_id,changes:{source_ids:[...]}}]只补字段，不必重发已保存项。重新考虑后不保存某草稿可给discard_reason；其他草稿成功不会自动取消失败项。"

RECALL_REMEMBER_SCHEMA = _schema("保存或修订自己的理解、表示、连接与注意，返回当前记忆地址。失败草稿可用retry修正。当前任务是为正在发生的交流提供背景，没有待完成的后台消息批次。", {
    key: value for key, value in REMEMBER_SCHEMA["parameters"]["properties"].items()
    if key in {"items", "retry", "recollection"}})


def tool_definitions(*, learning=False):
    definitions = {**TOOL_SCHEMAS, "remember": REMEMBER_SCHEMA if learning else RECALL_REMEMBER_SCHEMA}
    if learning:
        definitions.pop("main_context")
    else:
        definitions["complete"] = _schema("交付本轮语义背景并结束。可同时保存认识、连接与注意变化，不需要再调用模型读取保存回执。", {
            "background": TEXT, **RECALL_REMEMBER_SCHEMA["parameters"]["properties"]}, ("background",))
    return definitions


def _tool_set(*, learning=False):
    # Imported only when running under AstrBot; the core remains importable offline.
    from astrbot.core.agent.tool import FunctionTool, ToolSet
    definitions = tool_definitions(learning=learning)
    return ToolSet(tools=[FunctionTool(name=name, **definition) for name, definition in definitions.items()])
