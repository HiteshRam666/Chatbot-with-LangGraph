from langgraph.graph import START, END, StateGraph, MessagesState
from langgraph.graph.message import add_messages 
from typing import Annotated, Literal, TypedDict
from langchain_core.tools import tool 
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
import os 
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())
openai_api_key = os.environ["OPENAI_API_KEY"]


class Chatbot:
    def __init__(self):
        self.llm = ChatOpenAI()

    def call_tool(self):
        tool = TavilySearchResults(max_results=2)
        tools = [tool]
        self.tool_node = ToolNode(tools = [tool])
        self.llm_with_tool = self.llm.bind_tools(tools)
    
    def call_model(self, state: MessagesState):
        messages = state['messages']
        response = self.llm_with_tool.invoke(messages)
        return {'messages' : [response]}
    
    def router_function(self, state : MessagesState) -> Literal["tools", END]:
        messages = state['messages']
        last_message = messages[-1]
        if last_message.tool_calls:
            return 'tools'
        return END
    
    def __call__(self):
        self.call_tool() 
        workflow = StateGraph(MessagesState)
        workflow.add_node("agent", self.call_model)
        workflow.add_node("tools", self.tool_node)
        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges("agent", self.router_function, {"tools" : "tools", END : END})
        workflow.add_edge("tools", "agent")
        self.app = workflow.compile()
        return self.app
    
if __name__ == "__main__":
    mybot = Chatbot()
    workflow = mybot()
    response = workflow.invoke({"messages" : ["Who is the founded light bulb ?"]})
    print(response['messages'][-1].content)