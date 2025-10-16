import sys
import os
import nltk

# 判断是否打包环境
if getattr(sys, 'frozen', False):
    # EXE模式：使用临时解压目录或固定目录
    nltk_data_path = os.path.join(sys._MEIPASS, 'nltk_data')

    # 优先尝试写入程序所在目录（可永久保存）
    permanent_path = os.path.join(os.path.dirname(sys.executable), 'nltk_data')
    for path in [permanent_path, nltk_data_path]:
        if os.path.exists(path):
            nltk.data.path.append(path)
            break
else:
    # 开发模式：使用本地目录
    nltk.data.path.append('./nltk_data')

# 验证数据是否存在，不存在则触发下载
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', download_dir=permanent_path or nltk_data_path)

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import gradio as gr
import os, json

from html_string import main_html, plain_html
from upload_file import *
from create_kb import *
from chat import get_model_response
from mcp_server import call_tool, OscilloscopeControl
from check_quick_command import map_and_render


def _normalize(s: str) -> str:
    return " ".join(str(s or "").strip().casefold().split())


def _lookup_cmd(standercmd, user_text: str):
    # standercmd 可是 dict，也可能是带 find_command() 的对象
    if hasattr(standercmd, "find_command"):
        return standercmd.find_command(user_text)
    if isinstance(standercmd, dict):
        table = {_normalize(k): v for k, v in standercmd.items() if isinstance(v, list)}
        return table.get(_normalize(user_text))
    return None


def handle_submit(
        input_message, chatbot, selected_osc,
        model, temperature, max_tokens, history_round,
        knowledge_base, similarity_threshold, chunk_cnt
):
    # 1) 写历史 & 清空输入框
    cleared_input, chatbot = user(input_message, chatbot)
    # 2) 取本轮用户文本
    user_text = ""
    if chatbot and len(chatbot) > 0:
        user_text = chatbot[-1][0] or ""

    cmds = [":CURSor:ACTive OFF"]
    test_dict = {'scpi_cmd': ':CURSor:ACTive OFF'}

    # 3) 命中标准指令, 不走 LLM
    res = map_and_render(user_text)
    if res["route"] == "FAST":
        ok = True
        try:
            _ = call_tool("send_command", {"scpi_cmd": cmds}, selected_osc=selected_osc)
        except Exception:
            ok = False

        if ok:
            assistant_resp = f"用户输入「{user_text}」，命令「{cmds}」已经执行，请进行下一步操作。"
            right_panel = f"命令「{cmds}」已经执行，请进行下一步操作。"
        else:
            assistant_resp = f"用户输入「{cmds}」，SCPI 执行失败，请检查设备或指令。"
            right_panel = f"命令「{cmds}」执行失败，请重试或更换写法。"

        # 把助手回复写进当前轮对话（右槽）
        if chatbot and len(chatbot) > 0:
            chatbot[-1][-1] = assistant_resp
        else:
            chatbot = [[user_text, assistant_resp]]
        # 直接结束
        yield cleared_input, chatbot, right_panel

    # 4) 未命中, 走 LLM
    else:
        for chatbot, chunk_text in get_model_response(
                cleared_input, chatbot, model, temperature, max_tokens, history_round,
                knowledge_base, similarity_threshold, chunk_cnt, selected_osc
        ):
            yield cleared_input, chatbot, chunk_text


def user(user_message, history):
    print(user_message)
    # 把用户消息塞进 history，清空输入框
    return {'text': '', 'files': user_message['files']}, history + [[user_message['text'], None]]


#####################################
######       gradio界面       #######
#####################################

def get_chat_block():
    with gr.Blocks(theme=gr.themes.Base(), css=".gradio_container { background-color: #f0f0f0; }") as chat:
        gr.HTML(plain_html)
        with gr.Row():
            with gr.Column(scale=10):
                chatbot = gr.Chatbot(label="仪器智能体", height=750,
                                     avatar_images=("images/user.jpeg", "images/logo.png"))
                with gr.Row():
                    input_message = gr.MultimodalTextbox(label="请输入",
                                                         file_types=[".xlsx", ".csv", ".docx", ".pdf", ".txt"], scale=7)
                    clear_btn = gr.ClearButton(chatbot, input_message, scale=1)
            with gr.Column(scale=5):
                osc_app = OscilloscopeControl()
                devices = osc_app.refresh_visa_devices()
                if devices:
                    selected_osc = gr.Dropdown(choices=devices, label="选择示波器", interactive=True, value=devices[0],
                                               scale=2)
                knowledge_base = gr.Dropdown(choices=os.listdir(DB_PATH), label="加载知识库", interactive=True, scale=2)
                with gr.Accordion(label="召回文本段", open=False):
                    chunk_text = gr.Textbox(label="召回文本段", interactive=False, scale=5, lines=10)
                with gr.Accordion(label="模型设置", open=True):
                    model = gr.Dropdown(
                        choices=['qwen-max', 'qwen-plus', 'qwen-turbo', 'qwen2.5-7b-instruct', 'qwen2.5-32b-instruct'],
                        label="选择模型", interactive=True, value="qwen-max", scale=2)
                    temperature = gr.Slider(maximum=2, minimum=0, interactive=True, label="温度参数", step=0.01,
                                            value=0.0, scale=2)
                    max_tokens = gr.Slider(maximum=8000, minimum=0, interactive=True, label="最大回复长度", step=50,
                                           value=1024, scale=2)
                    history_round = gr.Slider(maximum=30, minimum=1, interactive=True, label="携带上下文轮数", step=1,
                                              value=3, scale=2)
                with gr.Accordion(label="RAG参数设置", open=True):
                    chunk_cnt = gr.Slider(maximum=20, minimum=1, interactive=True, label="选择召回片段数", step=1,
                                          value=5, scale=2)
                    similarity_threshold = gr.Slider(maximum=1, minimum=0, interactive=True, label="相似度阈值",
                                                     step=0.01, value=0.2, scale=2)

        # input_message.submit(fn=user, inputs=[input_message, chatbot], outputs=[input_message, chatbot],
        #                      queue=False).then(fn=get_model_response,
        #                                        inputs=[input_message, chatbot, model, temperature, max_tokens,
        #                                                history_round, knowledge_base, similarity_threshold, chunk_cnt,
        #                                                selected_osc], outputs=[chatbot, chunk_text]
        #                                        )

        input_message.submit(
            fn=handle_submit,
            inputs=[
                input_message, chatbot,  # 原始输入与历史
                selected_osc,  # 设备与标准指令表
                model, temperature, max_tokens,  # LLM 参数
                history_round, knowledge_base,  # 检索参数
                similarity_threshold, chunk_cnt  # 检索参数
            ],
            outputs=[input_message, chatbot, chunk_text],  # 第一个输出依然是“清空后的输入框”
            queue=False
        )

        chat.load(update_knowledge_base, [], knowledge_base)
        chat.load(clear_tmp)
    return chat


def get_upload_block():
    with gr.Blocks(theme=gr.themes.Base()) as upload:
        gr.HTML(plain_html)
        with gr.Tab("非结构化数据"):
            with gr.Accordion(label="新建类目", open=True):
                with gr.Column(scale=2):
                    unstructured_file = gr.Files(file_types=["pdf", "docx", "txt"])
                    with gr.Row():
                        new_label = gr.Textbox(label="类目名称", placeholder="请输入类目名称", scale=5)
                        create_label_btn = gr.Button("新建类目", variant="primary", scale=1)
            with gr.Accordion(label="管理类目", open=False):
                with gr.Row():
                    data_label = gr.Dropdown(choices=os.listdir(UNSTRUCTURED_FILE_PATH), label="管理类目",
                                             interactive=True, scale=8, multiselect=True)
                    delete_label_btn = gr.Button("删除类目", variant="stop", scale=1)
        with gr.Tab("结构化数据"):
            with gr.Accordion(label="新建数据表", open=True):
                with gr.Column(scale=2):
                    structured_file = gr.Files(file_types=["xlsx", "csv"])
                    with gr.Row():
                        new_label_1 = gr.Textbox(label="数据表名称", placeholder="请输入数据表名称", scale=5)
                        create_label_btn_1 = gr.Button("新建数据表", variant="primary", scale=1)
            with gr.Accordion(label="管理数据表", open=False):
                with gr.Row():
                    data_label_1 = gr.Dropdown(choices=os.listdir(STRUCTURED_FILE_PATH), label="管理数据表",
                                               interactive=True, scale=8, multiselect=True)
                    delete_data_table_btn = gr.Button("删除数据表", variant="stop", scale=1)
        delete_label_btn.click(delete_label, inputs=[data_label]).then(fn=update_label, outputs=[data_label])
        create_label_btn.click(fn=upload_unstructured_file, inputs=[unstructured_file, new_label]).then(fn=update_label,
                                                                                                        outputs=[
                                                                                                            data_label])
        delete_data_table_btn.click(delete_data_table, inputs=[data_label_1]).then(fn=update_datatable,
                                                                                   outputs=[data_label_1])
        create_label_btn_1.click(fn=upload_structured_file, inputs=[structured_file, new_label_1]).then(
            fn=update_datatable, outputs=[data_label_1])
        upload.load(update_label, [], data_label)
        upload.load(update_datatable, [], data_label_1)
    return upload


def get_knowledge_base_block():
    with gr.Blocks(theme=gr.themes.Base()) as knowledge:
        gr.HTML(plain_html)
        # 非结构化数据知识库
        with gr.Tab("非结构化数据"):
            with gr.Row():
                data_label_2 = gr.Dropdown(choices=os.listdir(UNSTRUCTURED_FILE_PATH), label="选择类目",
                                           interactive=True, scale=2, multiselect=True)
                knowledge_base_name = gr.Textbox(label="知识库名称", placeholder="请输入知识库名称", scale=2)
                create_knowledge_base_btn = gr.Button("确认创建知识库", variant="primary", scale=1)
        # 结构化数据知识库
        with gr.Tab("结构化数据"):
            with gr.Row():
                data_label_3 = gr.Dropdown(choices=os.listdir(STRUCTURED_FILE_PATH), label="选择数据表",
                                           interactive=True, scale=2, multiselect=True)
                knowledge_base_name_1 = gr.Textbox(label="知识库名称", placeholder="请输入知识库名称", scale=2)
                create_knowledge_base_btn_1 = gr.Button("确认创建知识库", variant="primary", scale=1)
        with gr.Row():
            knowledge_base = gr.Dropdown(choices=os.listdir(DB_PATH), label="管理知识库", interactive=True, scale=4)
            delete_db_btn = gr.Button("删除知识库", variant="stop", scale=1)
        create_knowledge_base_btn.click(fn=create_unstructured_db, inputs=[knowledge_base_name, data_label_2]).then(
            update_knowledge_base, outputs=[knowledge_base])
        delete_db_btn.click(delete_db, inputs=[knowledge_base]).then(update_knowledge_base, outputs=[knowledge_base])
        create_knowledge_base_btn_1.click(fn=create_structured_db, inputs=[knowledge_base_name_1, data_label_3]).then(
            update_knowledge_base, outputs=[knowledge_base])
        knowledge.load(update_knowledge_base, [], knowledge_base)
        knowledge.load(update_label, [], data_label_2)
        knowledge.load(update_datatable, [], data_label_3)
    return knowledge


app = FastAPI()


@app.get("/", response_class=HTMLResponse)
def read_main():
    html_content = main_html
    return HTMLResponse(content=html_content)


app = gr.mount_gradio_app(app, get_chat_block(), path="/chat")
app = gr.mount_gradio_app(app, get_upload_block(), path="/upload_data")
app = gr.mount_gradio_app(app, get_knowledge_base_block(), path="/create_knowledge_base")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=7866,
        reload=False,  # 必须关闭热重载
        workers=1,  # 单进程模式
        loop="asyncio"  # 明确指定事件循环
    )
