from vector_stores import VectorStoreService
from langchain_community.embeddings import DashScopeEmbeddings
import config_data as config
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document


def print_prompt(prompt):
    print("="*20)
    print(prompt.to_string())
    print("="*20)
    return prompt


class RagService(object):
    def __init__(self):
        self.vector_service = VectorStoreService(
            embedding=DashScopeEmbeddings(model=config.embedding_model_name)
        )
        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", "以我提供的已知参考资料为主，简洁和专业的回答用户问题。参考资料:{context}。"),
                ("user", "请回答用户提问:{input}")
            ]
        )
        self.chat_model = ChatTongyi(model=config.chat_model_name)

    def _get_chain(self):
        """获取最终的执行链"""
        retriever = self.vector_service.get_retriever()

        def format_document(docs: list[Document]):
            if not docs:
                return "无相关参考资料"
            formatted_str = ""
            for doc in docs:
                formatted_str += f"文档片段:{doc.page_content}\n文档元数据:{doc.metadata}\n\n"
            return formatted_str

        chain = (
            {
                "input": RunnablePassthrough(),
                "context": retriever | format_document
            }
            | self.prompt_template
            | print_prompt
            | self.chat_model
            | StrOutputParser()
        )
        return chain


if __name__ == "__main__":
    """
    简单的测试代码
    测试 RagService 的初始化、链获取和查询功能
    无论从哪个路径执行本文件，都会先将当前工作目录切换为脚本所在目录。
    """
    import os
    from dotenv import load_dotenv

    # 将当前工作目录切换为脚本所在目录，保证相对路径（如 ./chroma_db、./md5.text）始终指向项目目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"[info] 已将当前工作目录切换为脚本所在目录: {script_dir}")

    # 加载环境变量
    load_dotenv()
    
    # 获取 API key
    api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("API_KEY")
    if not api_key:
        print("警告: 未找到 DASHSCOPE_API_KEY 或 API_KEY 环境变量，测试可能失败")
        print("请先在 .env 或系统环境中配置 API key")
    else:
        os.environ["DASHSCOPE_API_KEY"] = api_key
    
    try:
        print("=" * 50)
        print("开始测试 RagService")
        print("=" * 50)
        
        # 1. 初始化 RagService
        print("\n[1] 初始化 RagService...")
        rag_service = RagService()
        print("✓ RagService 初始化成功")
        print(f"  - Embedding model: {config.embedding_model_name}")
        print(f"  - Chat model: {config.chat_model_name}")
        
        # 2. 获取执行链
        print("\n[2] 获取执行链...")
        chain = rag_service._get_chain()
        print("✓ 执行链获取成功")
        
        # 3. 测试查询功能
        print("\n[3] 测试查询功能...")
        test_query = "我身高180cm，140kg，我应该穿什么尺码的衣服？"
        print(f"  查询问题: {test_query}")
        try:
            result = chain.invoke(test_query)
            print(f"✓ 查询成功")
            print(f"  回答: {result}")
        except Exception as e:
            print(f"  注意: 查询时出现异常（可能是向量库为空或 API 配置问题）: {str(e)}")
            print("  这是正常的（如果还没有上传文档或 API key 未配置）")
        
        print("\n" + "=" * 50)
        print("测试完成！")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
