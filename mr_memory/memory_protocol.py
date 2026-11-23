"""Shared memory capabilities and purpose, independent of call scheduling."""
from copy import deepcopy
from urllib.parse import urlsplit

from .learning_writes import RETRY_SCHEMA

BASIC_RECALL_PROMPT = """你为 AstrBot 提供本群的记忆背景，帮助它接上真实群聊。当前模式只检索，不创建人物画像、学习反馈或发展新的长期连接。
结合当前消息、引用、近期对话和成员资料，自行决定检索词、语义查询、读取哪些原文，以及何时已有足够材料。旧记忆是检索入口，摘要可能有误；需要核对具体人物或经历时可打开对应原文及前后交流。UID 标明发言账号，引用作者、谈论对象与发言者各自保留，不能沿用相邻消息或图片的作者。
记录中 USER 是群友发言，BOT 是实际发送，SYSTEM 可能是 bot 生成稿或工具活动；同一请求的 generated/sent 不算两个独立事实。图片地址不等于看过图片。main_context 可读取主意识已有对话和工具，外部搜索、图像和行动由主意识负责。
成员目录的明确称呼与更正直接供本轮理解使用；群内简称和话中指代仍结合语境理解，重名不合并账号。不要把 bot 自己的旧猜测当成群友的确认。
完成时用 complete 的 background 字符串交付背景，或直接用纯文本交付；无须给纯文本再包 JSON。只提供主意识此刻缺少的群内来历，不代写回复，不替它决定外部行动。通常一两段，不附整篇历史机器人回答和思考报告。把握程度直接用自然语言表达。complete 的可选 references 数组选择已读原文的 {message_id} 或记忆的 {kind,id}，这些原文与真实作者会直接一同交给主意识，不必由你再转述。
没有需补充的背景可交付空字符串。工具返回的 source_ref/memory_ref 指本次此前已给出的相同材料，内容变化照常完整呈现。
"""

BASIC_CONSOLIDATION_PROMPT = """你为群聊机器人整理共同经历，建立以后能够主动检索、沿关联继续查阅的基础记忆图。
根据本批交流保存有用的事件、群友自述、具体事物及它们的联系；已有对应记忆时查阅后补充或修订，避免把同一件事重复建成互不相通的摘要。记什么、怎样组织、需要沿哪些线索查阅，由你根据语义决定，不要求每条消息生成一项记忆。
新建一项写 kind（例如 episode 或 semantic）、source_ids（实际支撑这项认识的原话编号）、title、content。用原文区分谁说的、说的是谁、后来怎样纠正。需要区分猜测与确认时写 belief；cues、connections、representation 只补正文尚未表达的线索、联系或结构。修订用已读的 kind/id，本批新对象可用 handle 相连。
搜索返回候选记忆的内容、版本、来源编号和关系地址。需要核对作者、原话或既有连接时，用 memory(kind,id) 打开对应的一项。群聊表中每行的 id 就是可引用的原文编号；同名不同账号在各自行内列出 UID，其他人的 UID 与角色见表内姓名对照。
UID 标明发言账号。结合引用作者、谈论对象及前后文理解归属；成员目录记录平台身份与指定称呼。写“某人要求／自述”时回看该人的措辞。机器人在回答或工具参数中补写的细节归到机器人；区分用户的要求、实际执行和完成回执。generated/sent 是同一次输出。图片路径能说明有图，画面细节需要实际观察依据。
本次工作以整理给定材料为目的，不启动独立反馈学习、持续关注或跨经历高级反思。原文中的明确纠正仍应体现在有关记忆中；模糊评价、反讽与身份推断保留语境和把握，不硬凑结论。
learning_task 列出本批 material_ids、实际提供的 offered_ids、仅供参考的 context_ids、已完成 completed_ids 及保存地址。输入中的 message_columns/message_rows/message_defaults 是完整原文表格，按表头读取作者、时间、正文和引用。
用 remember 或最后完整 JSON 保存 items 和 progress。已处理全部当前提供的材料时，progress 写 {"complete_offered":true}，系统展开 offered_ids，省去逐个抄写和核数；只完成部分时用 completed_ids 列出那部分。完成处理包括判断某些闲聊无需另存；无需穷尽所有关联才能完成本批。只保存未保存的内容，已保存回执地址可继续引用。finish=true 结束本批；尚未完成时 checkpoint 留下准确进度。
pending_items 是待修草稿，用 retry 的 pending_id 和 changes 补字段，或用 discard_reason 说明不再保存。草稿与材料进度分别保存。source_ref/memory_ref 指本次前文已给出的同版本字段，需要时可按地址重新打开。
queue 与 resource_state 表示各批次共享的资源；理解并保存后即可结束，不必用满额度。
"""

LEARNING_WRITE_FORMAT = """
同轮可以发起多个互不依赖的读取，取得结果后再据新线索继续。每轮都重新携带本批材料，合并独立读取可以减少重复输入。
message、context、speaker 的 columns/rows/defaults 均按表头读取完整记录。context_ref=true 复用此前相同 message_id 和 producer 的完整语境，角色、说话者和转发状态仍按那份记录理解。
reply_to.message_id、target_message_id 和 source_id 是已解析的 MR 原文编号，可交给 context 的 message_id 继续查看。
长工具结果的 result_preview 只展示开头，result_ref.message_id 指向完整记录；需要其细节时用 context(message_id,before=0,after=0) 读取。根据交流判断哪些外部材料值得进一步理解。
工具调用与最后 JSON 使用相同的记忆结构。connections 每个元素是带目标地址的对象：已存目标用 {kind,id,relation,context}，本批新目标用 {handle,relation,context}；关系描述写在 context，不能只写一个描述字符串。地址来自实际读取的记忆，新目标由同次提交的对象声明 handle。
新条目的非数字字符串 id 按临时 handle 理解，引用相同标签即可连接；数据库已有地址仍用整数 id。
修复旧草稿使用 retry:[{pending_id,changes:{...记忆字段...}}]；changes 不包 item。相互连接的待修草稿可以在同次 retry 一起提交并用 handle 互相引用。旧草稿里的错误结构是待修内容，不是保存示例。saved_connections 列出已经持久保存的连接；是否保存以写入回执为准，checkpoint 只记录思路。
"""
BASIC_CONSOLIDATION_PROMPT += LEARNING_WRITE_FORMAT

MESSAGE_CONTEXT_GUIDANCE = """
消息可附 message_context／source_contexts，标明插件提供的实际语境。sender_id/role 仍是发送账号；speaker 才是该段扮演角色，addressed_to 是对话对象，session_id/turn_id 标明归属。roleplay 里的角色生活、关系与剧情保留角色和会话语境，不归到运输消息的机器人或真实群友名下。群友参与了这场交流以及明确说出的现实信息仍可正常理解，不因带有 RP 语境而整段丢弃。forwarding=not_forwarded 表示未转给角色，不构成角色听过或回应过的经历。相同 utterance_id 的语音、字幕、原文和译文是同一表达；scene 是角色场景示意。它们都不增加独立事实支持。保留来源边界，具体意思、玩笑和现实自述仍结合语义判断。
"""
BASIC_RECALL_PROMPT += MESSAGE_CONTEXT_GUIDANCE
BASIC_CONSOLIDATION_PROMPT += MESSAGE_CONTEXT_GUIDANCE


def _schema(description: str, properties: dict, required: tuple = ()) -> dict:
    return {"description": description, "parameters": {"type": "object", "properties": properties,
            "required": list(required), "additionalProperties": False}}


TEXT = {"type": "string"}
INTEGER = {"type": "integer"}
ACCOUNT_ID = {"type": ["string", "integer"]}
TERMS = {"type": "array", "items": TEXT}
MEMORY_REF = {"type": "object", "properties": {"kind": TEXT, "id": INTEGER}, "required": ["kind", "id"]}
BELIEF = {"type": "object", "description": "由你按这项认识组织当前看法与适用范围，不要求填齐固定字段。把握针对具体解释，熟悉和自身复述不构成确认。"}
BACKGROUND = {"type": "string", "description": "直接写给主意识的背景正文，用自然语言保留语境与把握。"}
BACKGROUND_REFERENCES = {"type": "array", "items": {"anyOf": [MEMORY_REF,
    {"type": "object", "properties": {"message_id": INTEGER}, "required": ["message_id"]}]},
    "description": "可选的原始交流或当前记忆，按地址直接交付，避免经过另一次转述。"}
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
        "id": {"type": ["string", "integer"]}, "include_history": {"type": "boolean"},
        "include_sources": {"type": "boolean"}}, ("kind", "id")),
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
    "activity": _schema("某成员在Unix秒时间段的完整发言统计，返回总数、首末时间、每天与小时分布；可分析规律及估计作息，估计与观测分开。", {
        "account_id": TEXT, "start_at": INTEGER, "end_at": INTEGER},
        ("account_id", "start_at", "end_at")),
    "interaction": _schema("回看一次机器人互动：原请求、当时MR注入背景、主回答及工具与群友后续、曾形成的反思（包括当时已处理的）；detailed=true展开已记录的检索步骤。run_id加step只读指定步骤及其完整结果，可续读先前额度不足未交给模型的结果。读取记录不代表已确定因果。", {
        "request_id": TEXT, "run_id": INTEGER, "step": INTEGER, "detailed": {"type": "boolean"}}),
}


PROGRESS_SCHEMA = {"type": "object", "properties": {
    "completed_ids": {"type": "array", "items": INTEGER},
    "complete_offered": {"type": "boolean", "description": "已理解并处理本次提供的全部offered_ids时设为true，系统展开编号。只处理部分时填写completed_ids。"},
    "checkpoint": TEXT}, "additionalProperties": False}


REMEMBER_SCHEMA = _schema("保存新记忆或用kind和id修订已有记忆，并可同步保存本批处理进度。source_ids是证据；progress.completed_ids是已完整处理的材料，两者不同。仅保存进度或结束时可省略items，默认空列表。返回已保存记录和可复用的图节点id。finish=true表示本批材料已处理完，且不需再读取回执继续反思；材料进度独立保存，单项待修草稿不会阻止本批结束。", {
    "items": {"type": "array", "items": {"type": "object", "properties": {
        "kind": TEXT,
        "id": {"type": "integer", "description": "仅填写已读取的现有记忆地址。新建时省略id，用handle供同批其他对象引用；不要自行猜新编号。"},
        "source_ids": {"type": "array", "items": INTEGER,
                       "description": "本项认识依据的原始消息编号，供以后核对作者、语境与原话；填写本项实际依据，不等同于本批已处理编号。"},
        "handle": TEXT,
        "representation": {"type": "object", "description": "由你决定字段与层次的表示，随理解一起修订。"},
        "belief": BELIEF,
        "title": TEXT, "content": TEXT,
        "subject": {"type": "object", "properties": {"name": TEXT, "account_id": TEXT}},
        "source": {"type": "object", "properties": {"kind": TEXT, "id": INTEGER, "handle": TEXT, "node_id": INTEGER, "label": TEXT, "description": TEXT, "aliases": TERMS}},
        "target": {"type": "object", "properties": {"kind": TEXT, "id": INTEGER, "handle": TEXT, "node_id": INTEGER, "label": TEXT, "description": TEXT, "aliases": TERMS}},
        "relation": TEXT, "purpose": TEXT, "reason": TEXT,
        "cues": {"type": "array", "items": {"type": "object", "properties": {"cue": TEXT, "aspect": TEXT}, "required": ["cue", "aspect"]}},
        "connections": {"type": "array", "items": {"type": "object", "properties": {
            "kind": TEXT, "id": INTEGER, "handle": TEXT, "relation": TEXT, "context": TEXT, "purpose": TEXT, "connection_id": INTEGER,
            "target": {"type": "object", "description": "目标地址也可嵌套在这里，与平铺的kind/id或handle等价。",
                       "properties": {"kind": TEXT, "id": INTEGER, "handle": TEXT}},
            "note": {**TEXT, "description": "连接语境，与context等价。"},
            "belief": BELIEF, "reason": TEXT}, "required": ["relation"]}},
        "action": {"type": "string", "enum": ["revise", "withdraw"]},
        "started_at": INTEGER, "ended_at": INTEGER}, "required": ["kind"]}},
    "progress": PROGRESS_SCHEMA, "finish": {"type": "boolean"}, "retry": RETRY_SCHEMA})
WRITE_SEMANTICS = " 新建省略id，可用handle供同批source/target/connections引用；相互引用和自引用会连同对象一起保存，无须拆成多轮。修订用已读的kind/id，版本由系统管理，省略字段保留；withdraw撤回且保留历史。content写认识或关系语境，representation自定结构。connections从本项指向另一项，purpose=basis表示依据；connection_id可改已有边，新增不删除旧边。cues的cue/aspect说明自然线索及从它进入的方面。"
REMEMBER_SCHEMA["description"] = "保存本批的事件、语义记忆及有原文依据的关联，并记录处理进度。" + WRITE_SEMANTICS
# New memories state their original evidence explicitly. Revisions may omit it
# to preserve the existing sources, and derived memories can use an empty list.
REMEMBER_SCHEMA["parameters"]["properties"]["items"]["items"]["anyOf"] = [
    {"required": ["id"]}, {"required": ["source_ids"]}]


def is_poe_hydrogen(provider, model=None):
    config = getattr(provider, "provider_config", {})
    endpoint = config.get("api_base") or getattr(getattr(provider, "client", None), "base_url", "")
    if urlsplit(str(endpoint)).hostname != "api.poe.com":
        return False
    if model is None:
        model = provider.get_model()
    return str(model).rsplit("/", 1)[-1].lower() == "gemini-4-hydrogen"


def _hydrogen_remember_schema():
    # Poe accepts complete alternatives, but rejects required-only anyOf branches.
    definition = deepcopy(REMEMBER_SCHEMA)
    items = definition["parameters"]["properties"]["items"]
    item = items["items"]
    alternatives = item.pop("anyOf")
    items["items"] = {"anyOf": [
        {**item, "required": [*item["required"], *branch["required"]]} for branch in alternatives
    ]}
    return definition


HYDROGEN_REMEMBER_SCHEMA = _hydrogen_remember_schema()


def tool_definitions(*, learning=False, provider=None):
    names = ("search_messages", "search_memories", "memory", "semantic_search", "context", "member", "activity", "interaction", "graph", "navigate")
    definitions = {name: TOOL_SCHEMAS[name] for name in names}
    if learning:
        definitions["memory"] = _schema("打开完整记忆条目及来源编号。核对发言原文可传 include_sources=true，或用 context 按编号单独读取。", {
            **TOOL_SCHEMAS["memory"]["parameters"]["properties"],
            "include_sources": {"type": "boolean", "default": False}}, ("kind", "id"))
        definitions["remember"] = HYDROGEN_REMEMBER_SCHEMA if is_poe_hydrogen(provider) else REMEMBER_SCHEMA
    else:
        definitions["main_context"] = TOOL_SCHEMAS["main_context"]
        definitions["complete"] = _schema("交付本次检索背景并结束，不写入长期记忆；可以同时选择关键原话交给主意识。", {
            "background": BACKGROUND, "references": BACKGROUND_REFERENCES}, ("background",))
    return definitions


def _tool_set(*, learning=False, provider=None):
    # Imported only when running under AstrBot; the core remains importable offline.
    from astrbot.core.agent.tool import FunctionTool, ToolSet
    definitions = tool_definitions(learning=learning, provider=provider)
    return ToolSet(tools=[FunctionTool(name=name, **definition) for name, definition in definitions.items()])
