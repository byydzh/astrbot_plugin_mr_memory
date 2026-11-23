# MR 群聊记忆

MR 为 AstrBot 保存每个群的聊天记录。群友提起旧事时，回忆模型可以查原话、找相关经历，把选中的材料交给 AstrBot 的回复模型。默认在服务当地时间 00:00 至 06:00，整理模型把待处理的发言归成经历，也可以把新经历连到以前的记忆。

## 启用

1. 在 AstrBot 的插件配置中选择要记录的群，打开 `capture_enabled`。
2. 将 `subconscious_provider_id` 设为已配置的对话模型。回答前的回忆默认使用同一模型，可以通过 `local_serving_reader_provider_id` 单独指定。AstrBot 仍用群原有的回复模型发言。
3. 在插件管理页打开 **console**，选群查看收到的原话。查看一次回忆的运行记录，可以核对它查了什么、最终交给回复模型什么。

回忆和整理模型都需要支持原生函数调用。MR 支持 AstrBot 的 OpenAI Chat Completions、OpenAI Responses、Google Gemini 和 Anthropic Messages 接口，前后台可以选用不同接口。每轮分别应用 MR 的输出上限。Google 原生工具和思考级别沿用所选提供商的设置，收尾轮关闭工具，后台整理以 JSON 交付。直连 DeepSeek 时可分别设置前后台思考模式。其他自定义 AstrBot 提供商需要在公开调用接口声明输出上限参数，后台整理还需要 JSON 输出接口。

## 群员与数据

发 `/uid` 查看自己的资料，`/mrforget` 请求清除本人的记录。各群分别保存数据。更换 QQ 平台时，可在管理页的「平台迁移」预览匹配后接续旧群库。部署时保留插件数据目录。
