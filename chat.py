import os
from openai import OpenAI
from llama_index.core import StorageContext,load_index_from_storage,Settings
from llama_index.embeddings.dashscope import (
    DashScopeEmbedding,
    DashScopeTextEmbeddingModels,
    DashScopeTextEmbeddingType,
)
from llama_index.postprocessor.dashscope_rerank import DashScopeRerank
from create_kb import *
import anthropic
import json
import dashscope
from mcp_server import list_tools, call_tool
from mcp.server import Server

DB_PATH = "VectorStore"
TMP_NAME = "tmp_abcd"
API_KEY = os.getenv("DASHSCOPE_API_KEY")
if API_KEY is None:
    API_KEY = "sk-1600eff5b3ee4138929f1025f736a51d"
    os.environ["DASHSCOPE_API_KEY"] = API_KEY
dashscope.api_key = API_KEY
EMBED_MODEL = DashScopeEmbedding(
    model_name=DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V2,
    text_type=DashScopeTextEmbeddingType.TEXT_TYPE_DOCUMENT,
)
# 若使用本地嵌入模型，请取消以下注释：
# from langchain_community.embeddings import ModelScopeEmbeddings
# from llama_index.embeddings.langchain import LangchainEmbedding
# embeddings = ModelScopeEmbeddings(model_id="modelscope/iic/nlp_gte_sentence-embedding_chinese-large")
# EMBED_MODEL = LangchainEmbedding(embeddings)

# 设置嵌入模型
Settings.embed_model = EMBED_MODEL

def convertToAnthropicMessage(completion):
    openAiMessage = completion.choices[0].message
    anthropicMessage = anthropic.types.Message(
        id=completion.id,
        type="message",
        role=openAiMessage.role,
        content= [
        {
            "type": "text",
            "text": completion.choices[0].message.content or "",
            "citations": None,
        }
    ],
    model=completion.model,
    stop_reason = {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
        "content_filter": None
        }.get(completion.choices[0].finish_reason, None),
    stop_sequence=None,
    usage={
        "input_tokens": completion.usage.prompt_tokens if completion.usage else 0,
        "output_tokens": completion.usage.completion_tokens if completion.usage else 0,
        "cache_creation_input_tokens": None,
        "cache_read_input_tokens": None}
    )
    #转换调用工具消息格式
    if openAiMessage.tool_calls and len(completion.choices[0].message.tool_calls) > 0 :
        anthropicMessage.content.extend([anthropic.types.ToolUseBlock(**
        {
            "type": "tool_use",
            "id": tool_call.id,
            "name": tool_call.function.name,
            "input": json.loads(tool_call.function.arguments.replace("'", '"') if tool_call.function.arguments else "{}")
        })
        for tool_call in openAiMessage.tool_calls
    ])
    return anthropicMessage

# get_leaf_values = lambda d: [d] if not isinstance(d, dict) and not isinstance(d, list) else sum([get_leaf_values(v) for v in (d.values() if isinstance(d, dict) else d)], [])

def get_model_response(multi_modal_input,history,model,temperature,max_tokens,history_round,db_name,similarity_threshold,chunk_cnt,selected_osc):
    # prompt = multi_modal_input['text']
    prompt = history[-1][0]
    tmp_files = multi_modal_input['files']
    if os.path.exists(os.path.join("File",TMP_NAME)):
        db_name = TMP_NAME
    else:
        if tmp_files:
            create_tmp_kb(tmp_files)
            db_name = TMP_NAME
    # 获取index
    print(f"prompt:{prompt},tmp_files:{tmp_files},db_name:{db_name}")
    try:
        dashscope_rerank = DashScopeRerank(top_n=chunk_cnt,return_documents=True)
        storage_context = StorageContext.from_defaults(
            persist_dir=os.path.join(DB_PATH,db_name)
        )
        index = load_index_from_storage(storage_context)
        print("index获取完成")
        retriever_engine = index.as_retriever(
            similarity_top_k=20,
        )
        # 获取chunk
        retrieve_chunk = retriever_engine.retrieve(prompt)
        print(f"原始chunk为：{retrieve_chunk}")
        try:
            results = dashscope_rerank.postprocess_nodes(retrieve_chunk, query_str=prompt)
            print(f"rerank成功，重排后的chunk为：{results}")
        except:
            results = retrieve_chunk[:chunk_cnt]
            print(f"rerank失败，chunk为：{results}")
        chunk_text = ""
        chunk_show = ""
        for i in range(len(results)):
            if results[i].score >= similarity_threshold:
                chunk_text = chunk_text + f"## {i+1}:\n {results[i].text}\n"
                chunk_show = chunk_show + f"## {i+1}:\n {results[i].text}\nscore: {round(results[i].score,2)}\n"
        print(f"已获取chunk：{chunk_text}")
        prompt_template = f"请参考以下内容：{chunk_text}，以合适的语气回答用户的问题：{prompt}。如果参考内容中有图片链接也请直接返回。"
    except Exception as e:
        print(f"异常信息：{e}")
        prompt_template = prompt
        chunk_show = ""
    history[-1][-1] = ""
    client = OpenAI(
        api_key=API_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )                
    system_message = {'role': 'system', 'content': '你是示波器助手，帮助用户完成示波器的智能控制、分析。'}
    messages = []
    history_round = min(len(history),history_round)
    for i in range(history_round):
        messages.append({'role': 'user', 'content': history[-history_round+i][0]})
        messages.append({'role': 'assistant', 'content': history[-history_round+i][1]})
    messages.append({'role': 'user', 'content': prompt_template})
    messages = [system_message] + messages
    # completion = client.chat.completions.create(
    #     model=model,
    #     messages=messages,
    #     temperature=temperature,
    #     max_tokens=max_tokens,
    #     stream=True
    #     )
    # assistant_response = ""
    # for chunk in completion:
    #     assistant_response += chunk.choices[0].delta.content
    #     history[-1][-1] = assistant_response
    #     yield history,chunk_show

    server = Server("scpi")
    tools = list_tools()
    available_tools = [{ 
            "name": tool.name,
            "description": tool.description,
            "parameters": {k: v for k, v in tool.inputSchema.items() if k != "required"},
            "required": tool.inputSchema.get("required", []) 
        } for tool in tools]
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=False,
        tools=[
            {
                "type": "function",
                "function": available_tool
            }
            for available_tool in available_tools]
        # tool_choice={"type": "function", "function": {"name": "send_command"}}
    )

    response = convertToAnthropicMessage(completion)
    # print(response)

    if response.stop_reason == 'tool_use':
        for content in response.content:
            if content.type == 'tool_use':
                tool_name = content.name
                tool_args = content.input#{'scpi_cmd':get_leaf_values(content.input)}
                result = call_tool(tool_name, tool_args, selected_osc=selected_osc)
                messages_response = []
                messages_response.append({"role": "system", "content": "总结执行结果。"})
                messages_response.append({"role": "user", "content": result[0].text.encode().decode('unicode_escape')})

                completion_for_result = client.chat.completions.create(
                    model=model,#"qwen2.5-32b-instruct", # 此处以qwen-plus为例，可按需更换模型名称。模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
                    messages=messages_response,
                )
                response_for_result = convertToAnthropicMessage(completion_for_result)
                for content in response_for_result.content:
                    if content.type == 'text':
                        if hasattr(content, 'text') and content.text:
                            assistant_response = content.text
                            history[-1][-1] = f"调用工具： {tool_name}，执行命令： {tool_args}，" + assistant_response
    else:
        for content in response.content:
            if content.type == 'text':
                # final_text.append(content.text) #是否将大模型思考过程返回给最终用户
                if hasattr(content, 'text') and content.text:
                    assistant_response = content.text
                    history[-1][-1] = assistant_response
    yield history,chunk_show
