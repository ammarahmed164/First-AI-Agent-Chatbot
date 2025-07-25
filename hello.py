import os
import chainlit as cl
from dotenv import load_dotenv, find_dotenv
from agents import Agent, RunConfig, AsyncOpenAI, OpenAIChatCompletionsModel, Runner

load_dotenv(find_dotenv())
gemini_api_key = os.getenv("GEMINI_API_KEY") # jo api keys ".env" file me hain unhe hum yahan se le rahe hain.

provider = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=provider
)

run_config = RunConfig(
    model=model,
    model_provider=provider,
    tracing_disabled=True
)

# yahan hum ne Agent banaya hai jo ke Gemini API ko use karega. is agent ka name "agent1" hai.
agent1 = Agent(
    instructions="You are a helpful assistant that can answer questions",
    name="First AI Support Agent"
)

# Decorator to Maintain Conversation History:
@cl.on_chat_start
async def handle_chat_start():
    cl.user_session.set("history", []) # yahan hum ne 1 history ko "initialize/set" kiya hai jo ke abhi empty hai.
    await cl.Message(content="Hello! I'm the AI Support Agent. How can I help you?").send() # This message will be sent when the chat starts

# Decorator to Handle Incoming/new Messages and Maintain Conversation History:
@cl.on_message
async def handle_message(message: cl.Message):
    history = cl.user_session.get("history") # jb user message send karega to hum history ko get karenge. jo history abhi hum ne uper set ki thi. yahan just hum ne history ko call kia hai just..

    history.append({"role": "user", "content": message.content})# ab user ka content history me add ho gaya hai.

    # ✅ Build full conversation from history
    conversation = "\n".join([f"{h['role']}: {h['content']}" for h in history])

    # ab yahan hum ne conversation ko build kia hai jo ke history me se role aur content ko le kar banaya gaya hai.
    # overall yahan hum ne user ka proper message built kia hai jo ke AI ko send kia jayega.
    result = await Runner.run(
        agent1,                  # Yahan hum agent1 ko run kar rahain hain jo ke hum ne uper banaya tha.
        input=conversation,      # AI ko puri conversation (history) input me di ja rahi hai.Ab AI pura context samajh ke answer degi (sirf last message nahi).
        run_config=run_config,   #Ye AI ke settings/configurations pass kar raha hai (jo hum ne pehle banayi thi).
    )

    history.append({"role": "assistant", "content": result.final_output}) # ab yahan "agent/AI ka output/Answer" bhi history me add ho gaya hai.
    cl.user_session.set("history", history) # yahan hum ne history ko update kia hai jo history hum uper set/initialize kri thi.

    await cl.Message(content=result.final_output).send() # yahan Agent ne output send kia hai jo ke AI ka answer hai. Ye message user ko send/show kia ja raha hai.

