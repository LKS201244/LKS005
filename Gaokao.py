'''
考研择校助手 - 专业优化版
'''




from openai import OpenAI, Stream
import streamlit as st
from typing import Generator, Optional, List, Union
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionAssistantMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionFunctionMessageParam
)

# 全局配置
PROVINCES = ['北京', '天津', '河北', '山西', '内蒙古', '辽宁', '吉林', '黑龙江', '上海', '江苏', '浙江', '安徽', '福建', '江西', '山东', '河南', '湖北', '湖南', '广东', '广西', '海南', '重庆', '四川', '贵州', '云南', '西藏', '陕西', '甘肃', '青海', '宁夏', '新疆', '香港', '澳门', '台湾']
SUBJECT_TYPES = ["文科", "理工科", "经管类", "医学类", "艺术类"]
INTEREST_EXAMPLES = "如：计算机科学与技术、临床医学、金融学、法律硕士、教育学"

# 初始化全局变量
base_url: str = ""
model_name: str = ""

MessageParam = Union[
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionAssistantMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionFunctionMessageParam
]

def get_llm_response(
    client: OpenAI,
    model: str,
    user_prompt: str,
    system_prompt: Optional[str] = None,
    stream: bool = False
) -> Union[ChatCompletion, Stream[ChatCompletionChunk]]:
    """
    获取大语言模型响应

    Args:
        client: OpenAI客户端实例
        model: 使用的模型名称
        user_prompt: 用户提示词
        system_prompt: 系统提示词(可选)
        stream: 是否使用流式输出

    Returns:
        聊天完成对象或流式响应

    Raises:
        RuntimeError: 当API请求失败时
    """
    messages: List[ChatCompletionSystemMessageParam] = [
        dict["role": "user", "content": user_prompt]  # type: ignore
    ]

    if system_prompt:
        messages.insert(0, {"role": "system", "content": system_prompt})  # type: ignore

    try:
        return client.chat.completions.create(
            model=model,
            messages=messages,  # type: ignore
            stream=stream,
            temperature=0.7
        )
    except Exception as exc:
        raise RuntimeError(f"AI服务请求失败: {str(exc)}")

def get_advice(
    question: str,
    score: int,
    province: str,
    interests: str,
    subject_type: str
) -> Generator[str, None, None]:
    """
    智能生成考研择校建议（流式输出）

    Args:
        question: 用户咨询问题
        score: 初试总分(200-500)
        province: 目标院校省份
        interests: 研究方向
        subject_type: 学科门类

    Yields:
        建议内容字符串
    """
    try:
        if not 200 <= score <= 500:
            yield "⚠️ 请输入有效的考研初试分数（200-500之间）"
            return

        try:
            api_key = st.secrets['API_KEY']
            if not api_key.startswith(('sk-', 'lsk-')):
                raise ValueError("无效的API密钥格式")
        except Exception as e:
            yield f"⚠️ API密钥配置错误: {str(e)}"
            return

        prompt = f"""作为考研择校专家，请根据以下信息提供专业建议：

考生档案：
- 目标省份：{province}
- 初试总分：{score}分（{_get_score_level(score)}）
- 学科门类：{subject_type}
- 研究方向：{interests}

咨询问题：{question}

请按以下框架回答：
1. 【分数定位】该分数在目标院校层次的竞争力分析
2. 【院校推荐】建议报考的985/211/双非院校梯队
3. 【专业分析】研究方向对应的导师团队和学科排名
4. 【复试建议】该分数段的复试准备重点
5. 【调剂策略】如需调剂的备选方案"""

        client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )

        stream = get_llm_response(
            client=client,
            model=model_name,
            user_prompt=prompt,
            system_prompt="你是一名资深考研规划师，回答需：1.结合学科评估 2.关注导师资源 3.分点清晰",
            stream=True
        )

        for chunk in stream:
            content = chunk.choices[0].delta.content or ''
            yield content

    except Exception as exc:
        yield f"⚠️ 服务错误：{type(exc).__name__} - {str(exc)}"

def _get_score_level(score: int) -> str:
    """考研分数等级评估"""
    if score >= 400: return "🎖️ 顶尖水平（可冲985 top专业）"
    elif score >= 350: return "🏅 优秀水平（重点211优势专业）"
    elif score >= 300: return "📚 中等水平（普通一本主力区间）"
    return "📌 过线水平（需关注调剂机会）"

# ==============================================
# 页面界面配置
# ==============================================
st.set_page_config(
    page_title="智能考研择校助手",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="expanded"
)

# 侧边栏配置
with st.sidebar:
    st.header("🔧 系统配置", divider="rainbow")
    api_vendor = st.radio(
        "AI服务提供商",
        options=['OpenAI', 'deepseek'],
        index=0,
        horizontal=True
    )

    if api_vendor == 'OpenAI':
        base_url = st.selectbox(
            "API端点",
            options=[
                'https://api.openai.com/v1',
                'https://twapi.openai-hk.com/v1'
            ],
            index=0
        )
        model_options = ['gpt-4-turbo', 'gpt-4', 'gpt-3.5-turbo']
    else:
        base_url = 'https://api.deepseek.com'
        model_options = ['deepseek-chat', 'deepseek-v2']

    model_name = st.selectbox(
        "AI模型",
        options=model_options,
        index=0,
        help="更强大的模型能提供更精准的建议"
    )

    st.divider()
    st.header("📋 考生档案", divider="rainbow")
    user_score = st.slider(
        "初试总分",
        min_value=200,
        max_value=500,
        value=350,
        step=1,
        help="政治+英语+专业课总分（管理类联考等特殊考试除外）"
    )
    user_province = st.selectbox(
        "目标院校省份",
        options=PROVINCES,
        index=0,
        help="不同地区阅卷松紧度存在差异"
    )
    user_subject = st.radio(
        "学科门类",
        options=SUBJECT_TYPES,
        index=0,
        horizontal=True
    )
    user_interests = st.text_area(
        "研究方向",
        placeholder=INTEREST_EXAMPLES,
        help="建议细化到二级学科（如：人工智能方向）",
        height=100
    )

# 主界面
st.title("🎓 智能考研择校助手")
st.caption("AI驱动的考研咨询系统 | 数据仅供参考，请以各校研招网为准")

# 初始化对话
if 'messages' not in st.session_state:
    welcome_msg = """您好！我是考研择校助手小研，请按以下步骤操作：

1. 先在左侧填写【考生档案】
2. 然后提问，例如：
   - 350分能上哪些211计算机院校？
   - 推荐长三角地区的金融专硕院校
   - 这个分数报考985有希望吗？"""
    st.session_state.messages = [{"role": "ai", "content": welcome_msg}]

# 显示对话历史
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# 交互处理
if user_input := st.chat_input("输入您的问题..."):
    if not all([user_province, user_subject, user_interests]):
        st.error("请先完善左侧的考生基本信息！")
        st.stop()

    st.session_state.messages.append({"role": "human", "content": user_input})
    st.chat_message("human").write(user_input)

    with st.status("🔍 正在智能分析...", expanded=True) as status:
        try:
            advice_gen = get_advice(
                question=user_input,
                score=user_score,
                province=user_province,
                interests=user_interests,
                subject_type=user_subject
            )

            response = st.chat_message("ai").write_stream(advice_gen)
            st.session_state.messages.append({"role": "ai", "content": response})

            status.update(label="✅ 分析完成", state="complete", expanded=False)
        except Exception as exc:
            st.error(f"系统错误: {str(exc)}")
            st.session_state.messages.append({
                "role": "ai",
                "content": f"⚠️ 系统遇到错误，请稍后再试\n错误详情：{type(exc).__name__}"
            })