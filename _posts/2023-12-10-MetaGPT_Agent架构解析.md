# MetaGPT框架组件教程
## 1. Agent组件介绍 
### 1.1 Agent概念概述
在MetaGPT看来，我们把Agent想象成环境中的数字人，其中
Agent = 大语言模型（LLM） + 观察 + 思考 + 行动 + 记忆
这个公式概括了智能体的功能本质。为了理解每个组成部分，让我们将其与人类进行类比：  

1. 大语言模型（LLM）：LLM作为智能体的“大脑”部分，使其能够处理信息，从交互中学习，做出决策并执行行动。

2. 观察：这是智能体的感知机制，使其能够感知其环境。智能体可能会接收来自另一个智能体的文本消息、来自监视摄像头的视觉数据或来自客户服务录音的音频等一系列信号。这些观察构成了所有后续行动的基础。

3. 思考：思考过程涉及分析观察结果和记忆内容并考虑可能的行动。这是智能体内部的决策过程，其可能由LLM进行驱动。

4. 行动：这些是智能体对其思考和观察的显式响应。行动可以是利用 LLM 生成代码，或是手动预定义的操作，如阅读本地文件。此外，智能体还可以执行使用工具的操作，智能体还可以执行使用工具的操作，包括在互联网上搜索天气，使用计算器进行数学计算等。

5. 记忆：智能体的记忆存储过去的经验。这对学习至关重要，因为它允许智能体参考先前的结果并据此调整未来的行动。

在MetaGPT中定义的一个agent运行示例如下：

[![piRhAjs.png](https://z1.ax1x.com/2023/12/10/piRhAjs.png)](https://imgse.com/i/piRhAjs)

- 一个agent在启动后他会观察自己能获取到的信息，加入自己的记忆中
- 下一步进行思考，决定下一步的行动，也就是从Action1，Action2，Action3中选择执行的Action
- 决定行动后，紧接着就执行对应行动，得到这个环节的结果
而在MetaGPT内 Role 类是智能体的逻辑抽象。一个 Role 能执行特定的 Action，拥有记忆、思考并采用各种策略行动。基本上，它充当一个将所有这些组件联系在一起的凝聚实体。目前，让我们只关注一个执行动作的智能体，并看看如何实现一个最简单的 Agent

### 1.2 实现一个单动作Agent

下面将带领大家利用MetaGPT框架实现一个生成代码的Agent SimpleCoder 我们希望这个Agent 能够根据我们的需求来生成代码
要自己实现一个最简单的Role，只需要重写Role基类的 _init_ 与 _act 方法  

在 _init_ 方法中，我们需要声明 Agent 的name（名称）profile（类型）  

我们使用 self._init_action 函数为其配备期望的动作  

SimpleWriteCode 这个Action 应该能根据我们的需求生成我们期望的代码  

再_act方法中，我们需要编写智能体具体的行动逻辑，智能体将从最新的记忆中获取人类指令，运行配备的动作，MetaGPT将其作为待办事项 (self._rc.todo) 在幕后处理，最后返回一个完整的消息。  

#### 1.2.1  需求分析
要实现一个 SimpleCoder 我们需要分析这个Agent 它需要哪些能力  

[![piRhVun.png](https://z1.ax1x.com/2023/12/10/piRhVun.png)](https://imgse.com/i/piRhVun)  

首先我们需要让他接受用户的输入的需求，并记忆我们的需求，接着这个Agent它需要根据自己已知的信息和需求来编写我们需要的代码。  

#### 1.2.2 编写SimpleWriteCode动作
在 MetaGPT 中，类 Action 是动作的逻辑抽象。用户可以通过简单地调用 self._aask 函数令 LLM 赋予这个动作能力，即这个函数将在底层调用 LLM api。  

下面是实现SimpleWriteCode的具体代码：  

```python
from metagpt.actions import Action

class SimpleWriteCode(Action):

    PROMPT_TEMPLATE = """
    Write a python function that can {instruction} and provide two runnnable test cases.
    Return ```python your_code_here ``` with NO other texts,
    your code:
    """

    def __init__(self, name="SimpleWriteCode", context=None, llm=None):
        super().__init__(name, context, llm)

    async def run(self, instruction: str):

        prompt = self.PROMPT_TEMPLATE.format(instruction=instruction)

        rsp = await self._aask(prompt)

        code_text = SimpleWriteCode.parse_code(rsp)

        return code_text

    @staticmethod
    def parse_code(rsp):
        pattern = r'```python(.*)```'
        match = re.search(pattern, rsp, re.DOTALL)
        code_text = match.group(1) if match else rsp
        return code_text
```

在我们的场景中，我们定义了一个 SimpleWriteCode 类，它继承自 Action 类，我们重写了 __init__方法与 run 方法 __init__方法用来初始化这个Action，而 run 方法决定了我们对传入的内容到底要做什么样的处理  

在__init__方法中，我们声明了这个类要使用的 llm ，这个动作的名称，以及行动前的一些前置知识（context），这里 context 为空  

```python
def __init__(self, name="SimpleWriteCode", context=None, llm=None):
        super().__init__(name, context, llm)  
```
在 run 方法中，我们需要声明当采取这个行动时，我们要对传入的内容做什么样的处理，在 SimpleWriteCode 类中，我们应该传入：“请你帮我写一个XXX的代码” 这样的字符串，也就是用户的输入， run 方法需要对它进行处理，把他交给 llm ，等到 llm 返回生成结果后，我们再取出其中的代码部分返回。

我们写好了一个提示词模板，将用户输入嵌入模板中  

```python
PROMPT_TEMPLATE = """
    Write a python function that can {instruction} and provide two runnnable test cases.
    Return ```python your_code_here ``` with NO other texts,
    your code:
    """
prompt = self.PROMPT_TEMPLATE.format(instruction=instruction)
```
接着我们让大模型为我们生成回答  

```python
rsp = await self._aask(prompt)
生成回答后，我们利用正则表达式提取其中的code部分，llm在返回给我们代码时通常用下面的形式返回  
```

```python
code内容
```

对应的正则提取内容如下：  

parse_code方法使用正则表达式来匹配用户输入的代码文本。它会查找以"python"开头，""结尾的代码块，并提取其中的代码内容。如果找到匹配的代码块，则返回提取的代码内容；否则，返回原始的用户输入。  
```python
@staticmethod
def parse_code(rsp):
    pattern = r'```python(.*)```'
    match = re.search(pattern, rsp, re.DOTALL)
    code_text = match.group(1) if match else rsp
    return code_text
```
最后将代码内容返回  
至此我们就完成了这样一个编写代码的动作。

#### 1.2.3 设计SimpleCoder角色

在此之前我们需要简单介绍一下 Message，在MetaGPT中，Message 类是最基本的信息类型，Message 的基本组成如下
[![piRhkcj.png](https://z1.ax1x.com/2023/12/10/piRhkcj.png)](https://imgse.com/i/piRhkcj)
在本章节的学习中我们只涉及 content  role  cause_by ，除了content外，其他内容都是可选的
他们分别代表信息内容，发出信息的角色，以及是哪个动作导致产生的message

在编写完SimpleWriteCode动作后，我相信大家还有很多疑惑，比如如何调用这个动作？怎样把用户输入的内容传递给这个动作？
这部分内容我们都会在设计SimpleCoder角色的时候解决

```python
class SimpleCoder(Role):
    def __init__(
        self,
        name: str = "Alice",
        profile: str = "SimpleCoder",
        **kwargs,
    ):
        super().__init__(name, profile, **kwargs)
        self._init_actions([SimpleWriteCode])

    async def _act(self) -> Message:
        logger.info(f"{self._setting}: ready to {self._rc.todo}")
        todo = self._rc.todo # todo will be SimpleWriteCode()

        msg = self.get_memories(k=1)[0] # find the most recent messages

        code_text = await todo.run(msg.content)
        msg = Message(content=code_text, role=self.profile, cause_by=type(todo))

        return msg
```

前面我们已经提到过实现一个最简单的Role，只需要重写Role基类的_init_ 与_act 方法

__init__方法用来初始化这个Action，而_act方法决定了当这个角色行动时它的具体行动逻辑  

我们在__init__ 方法中声明了这个Role的name（昵称），profile（人设），以及我们为他配备了我们之前写好的动作 SimpleWriteCode  
```python
def __init__(
        self,
        name: str = "Alice",
        profile: str = "SimpleCoder",
        **kwargs,
    ):
        super().__init__(name, profile, **kwargs)
        self._init_actions([SimpleWriteCode])
```
配备好之后，我们定义的行动SimpleWriteCode就会被加入到代办self._rc.todo中

在_act方法中，我们就会要求我们的智能体来执行这个动作，也就是我们需要调用todo.run()方法

```python
async def _act(self) -> Message:
    logger.info(f"{self._setting}: ready to {self._rc.todo}")
    todo = self._rc.todo  # todo will be SimpleWriteCode()
```

另外我们在 前面的action 中提到了，当action调用时，我们需要获取用户的输入来作为instruction传递给action，这里就涉及到我们该如何获取用户之前传递给agent的信息，在MetaGPT中，当用户与Agent交互时，所有的内容都会被存储在其自有的Memory中  

在MetaGPT中，Memory类是智能体的记忆的抽象。当初始化时，Role初始化一个Memory对象作为self._rc.memory属性，它将在之后的_observe中存储每个Message，以便后续的检索。简而言之，Role的记忆是一个含有Message的列表。  

当需要获取记忆时（获取LLM输入的上下文），我们可以使用self.get_memories。函数定义如下：  

```python
def get_memories(self, k=0) -> list[Message]:
    """A wrapper to return the most recent k memories of this role, return all when k=0"""
    return self._rc.memory.get(k=k)
```
在SimpleCoder中，我们只需要获取最近的一条记忆，也就是用户下达的需求，将它传递给action即可

```python
msg = self.get_memories(k=1)[0]  # find the most recent messages
code_text = await todo.run(msg.content)
```
然后我们就将拿到大模型给我们的输出啦，最后我们将拿到的信息封装为MetaGPT中通信的基本格式 Message 返回，
这样，我们就实现了一个简单的单动作Agent

#### 1.2.4 运行SimpleCoder角色
接下来你只需要初始化它并使用一个起始消息运行它。
```python
import asyncio

async def main():
    msg = "write a function that calculates the sum of a list"
    role = SimpleCoder()
    logger.info(msg)
    result = await role.run(msg)
    logger.info(result)

asyncio.run(main())
```

完整代码如下：

```python
import re
import asyncio
from metagpt.actions import Action
from metagpt.roles import Role
from metagpt.schema import Message
from metagpt.logs import logger

class SimpleWriteCode(Action):

    PROMPT_TEMPLATE = """
    Write a python function that can {instruction} and provide two runnnable test cases.
    Return ```python your_code_here ``` with NO other texts,
    your code:
    """

    def __init__(self, name="SimpleWriteCode", context=None, llm=None):
        super().__init__(name, context, llm)

    async def run(self, instruction: str):

        prompt = self.PROMPT_TEMPLATE.format(instruction=instruction)

        rsp = await self._aask(prompt)

        code_text = SimpleWriteCode.parse_code(rsp)

        return code_text

    @staticmethod
    def parse_code(rsp):
        pattern = r'```python(.*)```'
        match = re.search(pattern, rsp, re.DOTALL)
        code_text = match.group(1) if match else rsp
        return code_text

class SimpleCoder(Role):
    def __init__(
        self,
        name: str = "Alice",
        profile: str = "SimpleCoder",
        **kwargs,
    ):
        super().__init__(name, profile, **kwargs)
        self._init_actions([SimpleWriteCode])

    async def _act(self) -> Message:
        logger.info(f"{self._setting}: ready to {self._rc.todo}")
        todo = self._rc.todo  # todo will be SimpleWriteCode()

        msg = self.get_memories(k=1)[0]  # find the most recent messages

        code_text = await todo.run(msg.content)
        msg = Message(content=code_text, role=self.profile,
                      cause_by=type(todo))

        return msg

async def main():
    msg = "write a function that calculates the sum of a list"
    role = SimpleCoder()
    logger.info(msg)
    result = await role.run(msg)
    logger.info(result)

asyncio.run(main())
```

## 1.3 实现一个多动作Agent

我们注意到一个智能体能够执行一个动作，但如果只有这些，实际上我们并不需要一个智能体。通过直接运行动作本身，我们可以得到相同的结果。智能体的力量，或者说Role抽象的惊人之处，在于动作的组合（以及其他组件，比如记忆，但我们将把它们留到后面的部分）。通过连接动作，我们可以构建一个工作流程，使智能体能够完成更复杂的任务。

### 1.3.1 需求分析
[![piRhF3Q.png](https://z1.ax1x.com/2023/12/10/piRhF3Q.png)](https://imgse.com/i/piRhF3Q)
假设现在我们不仅希望用自然语言编写代码，而且还希望生成的代码立即执行。一个拥有多个动作的智能体可以满足我们的需求。让我们称之为RunnableCoder，一个既写代码又立即运行的Role。我们需要两个Action：SimpleWriteCode 和 SimpleRunCode

### 1.3.2 编写SimpleWriteCode动作  

这部分与我们在前文中讲到的基本一致

```python
class SimpleWriteCode(Action):

    PROMPT_TEMPLATE = """
    Write a python function that can {instruction} and provide two runnnable test cases.
    Return ```python your_code_here ``` with NO other texts,
    your code:
    """

    def __init__(self, name="SimpleWriteCode", context=None, llm=None):
        super().__init__(name, context, llm)

    async def run(self, instruction: str):

        prompt = self.PROMPT_TEMPLATE.format(instruction=instruction)

        rsp = await self._aask(prompt)

        code_text = SimpleWriteCode.parse_code(rsp)

        return code_text

    @staticmethod
    def parse_code(rsp):
        pattern = r'```python(.*)```'
        match = re.search(pattern, rsp, re.DOTALL)
        code_text = match.group(1) if match else rsp
        return code_text
```

### 1.3.3 编写 SimpleRunCode 动作

从概念上讲，一个动作可以利用LLM，也可以在没有LLM的情况下运行。在SimpleRunCode的情况下，LLM不涉及其中。我们只需启动一个子进程来运行代码并获取结果 

在Python中，我们通过标准库中的subprocess包来fork一个子进程，并运行一个外部的程序。

subprocess包中定义有数个创建子进程的函数，这些函数分别以不同的方式创建子进程，所以我们可以根据需要来从中选取一个使用。

第一个进程是你的Python程序本身，它执行了包含 SimpleRunCode 类定义的代码。第二个进程是由 subprocess.run 创建的，它执行了 python3 -c 命令，用于运行 code_text 中包含的Python代码。这两个进程相互独立，通过 subprocess.run 你的Python程序可以启动并与第二个进程进行交互，获取其输出结果。

```python
class SimpleRunCode(Action):
    def __init__(self, name="SimpleRunCode", context=None, llm=None):
        super().__init__(name, context, llm)

    async def run(self, code_text: str):
        # 在Windows环境下，result可能无法正确返回生成结果，在windows中在终端中输入python3可能会导致打开微软商店
        result = subprocess.run(["python3", "-c", code_text], capture_output=True, text=True)
        # 采用下面的可选代码来替换上面的代码
        # result = subprocess.run(["python", "-c", code_text], capture_output=True, text=True)
        # import sys
        # result = subprocess.run([sys.executable, "-c", code_text], capture_output=True, text=True)
        code_result = result.stdout
        logger.info(f"{code_result=}")
        return code_result
```

### 1.3.4 定义 RunnableCoder 角色

与定义单一动作的智能体没有太大不同！让我们来映射一下：

1. 用 self._init_actions 初始化所有 Action

2. 指定每次 Role 会选择哪个 Action。我们将 react_mode 设置为 "by_order"，这意味着 Role 将按照 self._init_actions 中指定的顺序执行其能够执行的 Action。在这种情况下，当 Role 执行 _act 时，self._rc.todo 将首先是 SimpleWriteCode，然后是 SimpleRunCode。

3. 覆盖 _act 函数。Role 从上一轮的人类输入或动作输出中检索消息，用适当的 Message 内容提供当前的 Action (self._rc.todo)，最后返回由当前 Action 输出组成的 Message。

这里我们用Role类的 _set_react_mode 方法来设定我们action执行的先后顺序，事实上Role基类中还包含了很多有用的方法，你可以自己查看它的定义，在后面的章节内容中，我们也将一步一步揭开他们的面纱。

```python
class RunnableCoder(Role):
    def __init__(
        self,
        name: str = "Alice",
        profile: str = "RunnableCoder",
        **kwargs,
    ):
        super().__init__(name, profile, **kwargs)
        self._init_actions([SimpleWriteCode, SimpleRunCode])
        self._set_react_mode(react_mode="by_order")

    async def _act(self) -> Message:
        logger.info(f"{self._setting}: 准备 {self._rc.todo}")
        # 通过在底层按顺序选择动作
        # todo 首先是 SimpleWriteCode() 然后是 SimpleRunCode()
        todo = self._rc.todo

        msg = self.get_memories(k=1)[0] # 得到最相似的 k 条消息
        result = await todo.run(msg.content)

        msg = Message(content=result, role=self.profile, cause_by=type(todo))
        self._rc.memory.add(msg)
        return msg
```

### 1.3.5 运行 RunnableCoder 角色

这部分与SimpleCoder基本一致，只需要修改我们使用的role为RunnableCoder

```python
import asyncio

async def main():
    msg = "write a function that calculates the sum of a list"
    role = RunnableCoder()
    logger.info(msg)
    result = await role.run(msg)
    logger.info(result)

asyncio.run(main())
```
完整代码如下：

```python
import os
import re
import subprocess
import asyncio

import fire
import sys
from metagpt.llm import LLM
from metagpt.actions import Action
from metagpt.roles import Role
from metagpt.schema import Message
from metagpt.logs import logger

class SimpleWriteCode(Action):

    PROMPT_TEMPLATE = """
    Write a python function that can {instruction} and provide two runnnable test cases.
    Return ```python your_code_here ``` with NO other texts,
    your code:
    """

    def __init__(self, name: str = "SimpleWriteCode", context=None, llm: LLM = None):
        super().__init__(name, context, llm)

    async def run(self, instruction: str):

        prompt = self.PROMPT_TEMPLATE.format(instruction=instruction)

        rsp = await self._aask(prompt)

        code_text = SimpleWriteCode.parse_code(rsp)

        return code_text

    @staticmethod
    def parse_code(rsp):
        pattern = r'```python(.*)```'
        match = re.search(pattern, rsp, re.DOTALL)
        code_text = match.group(1) if match else rsp
        return code_text

class SimpleRunCode(Action):
    def __init__(self, name: str = "SimpleRunCode", context=None, llm: LLM = None):
        super().__init__(name, context, llm)

    async def run(self, code_text: str):
        result = subprocess.run([sys.executable, "-c", code_text], capture_output=True, text=True)
        code_result = result.stdout
        logger.info(f"{code_result=}")
        return code_result

class RunnableCoder(Role):
    def __init__(
        self,
        name: str = "Alice",
        profile: str = "RunnableCoder",
        **kwargs,
    ):
        super().__init__(name, profile, **kwargs)
        self._init_actions([SimpleWriteCode, SimpleRunCode])
        self._set_react_mode(react_mode="by_order")

    async def _act(self) -> Message:
        logger.info(f"{self._setting}: ready to {self._rc.todo}")
        # By choosing the Action by order under the hood
        # todo will be first SimpleWriteCode() then SimpleRunCode()
        todo = self._rc.todo

        msg = self.get_memories(k=1)[0] # find the most k recent messagesA
        result = await todo.run(msg.content)

        msg = Message(content=result, role=self.profile, cause_by=type(todo))
        self._rc.memory.add(msg)
        return msg

async def main():
    msg = "write a function that calculates the sum of a list"
    role = RunnableCoder()
    logger.info(msg)
    result = await role.run(msg)
    logger.info(result)

asyncio.run(main())
```

## 1.4 实现一个更复杂的Agent：技术文档助手

在前文中我们已经介绍了如何实现一个简单的agent帮我们生成代码并执行代码，下面我们将带领大家实现更复杂的agent，并向大家展示MetaGPT中关于agent的更多设计细节

现在试着想想怎么让大模型为我们写一篇技术文档？

可能想到的是，我们告诉大模型：“请帮我生成关于Mysql的技术文档”，他可能很快地就能帮你完成这项任务，但是受限于大模型自身的token限制，我们无法实现让他一次性就输出我们希望的一个完整的技术文档。

当然我们可以将我们的技术文档拆解成一个一个很小的需求，然后一个一个的提问，但是这样来说不仅费时，而且还需要人工一直去跟他交互，非常的麻烦，下面我们就将利用MetaGPT框架来解决这个问题

我们利用上文中提到的agent框架来拆解我们的需求  

### 1.4.1 需求分析

因为token限制的原因，我们先通过 LLM 大模型生成教程的目录，再对目录按照二级标题进行分块，对于每块目录按照标题生成详细内容，最后再将标题和内容进行拼接，解决 LLM 大模型长文本的限制问题。
[![piRhZBq.png](https://z1.ax1x.com/2023/12/10/piRhZBq.png)](https://imgse.com/i/piRhZBq)

### 1.4.2 编写 WriteDirectory 动作

我们先来实现根据用户需求生成文章大纲的代码

```python
class WriteDirectory(Action):
    """Action class for writing tutorial directories.

    Args:
        name: The name of the action.
        language: The language to output, default is "Chinese".
    """

    def __init__(self, name: str = "", language: str = "Chinese", *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self.language = language

    async def run(self, topic: str, *args, **kwargs) -> Dict:
        """Execute the action to generate a tutorial directory according to the topic.

        Args:
            topic: The tutorial topic.

        Returns:
            the tutorial directory information, including {"title": "xxx", "directory": [{"dir 1": ["sub dir 1", "sub dir 2"]}]}.
        """
        COMMON_PROMPT = """
        You are now a seasoned technical professional in the field of the internet. 
        We need you to write a technical tutorial with the topic "{topic}".
        """

        DIRECTORY_PROMPT = COMMON_PROMPT + """
        Please provide the specific table of contents for this tutorial, strictly following the following requirements:
        1. The output must be strictly in the specified language, {language}.
        2. Answer strictly in the dictionary format like {{"title": "xxx", "directory": [{{"dir 1": ["sub dir 1", "sub dir 2"]}}, {{"dir 2": ["sub dir 3", "sub dir 4"]}}]}}.
        3. The directory should be as specific and sufficient as possible, with a primary and secondary directory.The secondary directory is in the array.
        4. Do not have extra spaces or line breaks.
        5. Each directory title has practical significance.
        """
        prompt = DIRECTORY_PROMPT.format(topic=topic, language=self.language)
        resp = await self._aask(prompt=prompt)
        return OutputParser.extract_struct(resp, dict)
```

基本就是我们把自己的需求放入我们准备好的提示词模板里，询问大模型得到结果，然后我们对得到的内容做一个解析。

```python
def extract_struct(cls, text: str, data_type: Union[type(list), type(dict)]) -> Union[list, dict]:
    """Extracts and parses a specified type of structure (dictionary or list) from the given text.
    The text only contains a list or dictionary, which may have nested structures.

    Args:
        text: The text containing the structure (dictionary or list).
        data_type: The data type to extract, can be "list" or "dict".

    Returns:
        - If extraction and parsing are successful, it returns the corresponding data structure (list or dictionary).
        - If extraction fails or parsing encounters an error, it throw an exception.

    Examples:
        >>> text = 'xxx [1, 2, ["a", "b", [3, 4]], {"x": 5, "y": [6, 7]}] xxx'
        >>> result_list = OutputParser.extract_struct(text, "list")
        >>> print(result_list)
        >>> # Output: [1, 2, ["a", "b", [3, 4]], {"x": 5, "y": [6, 7]}]

        >>> text = 'xxx {"x": 1, "y": {"a": 2, "b": {"c": 3}}} xxx'
        >>> result_dict = OutputParser.extract_struct(text, "dict")
        >>> print(result_dict)
        >>> # Output: {"x": 1, "y": {"a": 2, "b": {"c": 3}}}
    """
    # Find the first "[" or "{" and the last "]" or "}"
    start_index = text.find("[" if data_type is list else "{")
    end_index = text.rfind("]" if data_type is list else "}")

    if start_index != -1 and end_index != -1:
        # Extract the structure part
        structure_text = text[start_index : end_index + 1]

        try:
            # Attempt to convert the text to a Python data type using ast.literal_eval
            result = ast.literal_eval(structure_text)

            # Ensure the result matches the specified data type
            if isinstance(result, list) or isinstance(result, dict):
                return result

            raise ValueError(f"The extracted structure is not a {data_type}.")

        except (ValueError, SyntaxError) as e:
            raise Exception(f"Error while extracting and parsing the {data_type}: {e}")
    else:
        logger.error(f"No {data_type} found in the text.")
        return [] if data_type is list else {}
```
注释里给了解析的example，这里再提一下

```shell
>>> text = 'xxx [1, 2, ["a", "b", [3, 4]], {"x": 5, "y": [6, 7]}] xxx'
>>> result_list = OutputParser.extract_struct(text, "list")
>>> print(result_list)
>>> # Output: [1, 2, ["a", "b", [3, 4]], {"x": 5, "y": [6, 7]}]

>>> text = 'xxx {"x": 1, "y": {"a": 2, "b": {"c": 3}}} xxx'
>>> result_dict = OutputParser.extract_struct(text, "dict")
>>> print(result_dict)
>>> # Output: {"x": 1, "y": {"a": 2, "b": {"c": 3}}}
```

这样我们就将大模型输出的目录结构转为了可解析的字典对象，这里以写一篇Mysql教程文档为例，它的输出就如下：

```python
{'title': 'MySQL 教程', 'directory': [{'MySQL 简介': []}, {'安装与配置': ['安装MySQL', '配置MySQL']}, {'基本操作': ['创建数据库', '创建表', '插入数据', '查询数据', '更新数据', '删除数据']}, {'高级操作': ['索引', '约束', '连接查询', '子查询', '事务', '视图']}, {'备份与恢复': ['备份数据库', '恢复数据库']}, {'性能优化': ['优化查询语句', '优化表结构', '缓存配置']}, {'常见问题': ['连接问题', '权限问题', '性能问题']}]}
```

拿到目录后我们就需要根据每个章节的内容生成章节内容了

### 1.4.3 编写 WriteContent 动作

接下来我们需要根据传入的子标题来生成内容

```python
class WriteContent(Action):
    """Action class for writing tutorial content.

    Args:
        name: The name of the action.
        directory: The content to write.
        language: The language to output, default is "Chinese".
    """

    def __init__(self, name: str = "", directory: str = "", language: str = "Chinese", *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self.language = language
        self.directory = directory

    async def run(self, topic: str, *args, **kwargs) -> str:
        """Execute the action to write document content according to the directory and topic.

        Args:
            topic: The tutorial topic.

        Returns:
            The written tutorial content.
        """
        COMMON_PROMPT = """
        You are now a seasoned technical professional in the field of the internet. 
        We need you to write a technical tutorial with the topic "{topic}".
        """
        CONTENT_PROMPT = COMMON_PROMPT + """
        Now I will give you the module directory titles for the topic. 
        Please output the detailed principle content of this title in detail. 
        If there are code examples, please provide them according to standard code specifications. 
        Without a code example, it is not necessary.

        The module directory titles for the topic is as follows:
        {directory}

        Strictly limit output according to the following requirements:
        1. Follow the Markdown syntax format for layout.
        2. If there are code examples, they must follow standard syntax specifications, have document annotations, and be displayed in code blocks.
        3. The output must be strictly in the specified language, {language}.
        4. Do not have redundant output, including concluding remarks.
        5. Strict requirement not to output the topic "{topic}".
        """
        prompt = CONTENT_PROMPT.format(
            topic=topic, language=self.language, directory=self.directory)
        return await self._aask(prompt=prompt)
```

这里我们直接根据传入的子标题内容调用大模型生成回答即可

### 1.4.4 编写 TutorialAssistant 角色

编写完动作后，还有一个问题需要我们解决，按照我们的设计，大模型应该先调用WriteDirectory 动作去生成大纲，然后根据大纲的内容去生成对应的内容，我们很难把这整个流程都设计为固定流程，因为当我们需要生成的内容变化时，大纲的结构也会随之变化，当然你也可以在提示词中限制大纲的结构，但是这种解决方法无疑不够优雅而且灵活欠佳，这一章节内，我们将为你展示MetaGPT是如何组织Action的行动路线的

我们依然先重写_init_方法来初始化我们的角色

```python
class TutorialAssistant(Role):
    """Tutorial assistant, input one sentence to generate a tutorial document in markup format.

    Args:
        name: The name of the role.
        profile: The role profile description.
        goal: The goal of the role.
        constraints: Constraints or requirements for the role.
        language: The language in which the tutorial documents will be generated.
    """

    def __init__(
        self,
        name: str = "Stitch",
        profile: str = "Tutorial Assistant",
        goal: str = "Generate tutorial documents",
        constraints: str = "Strictly follow Markdown's syntax, with neat and standardized layout",
        language: str = "Chinese",
    ):
        super().__init__(name, profile, goal, constraints)
        self._init_actions([WriteDirectory(language=language)])
        self.topic = ""
        self.main_title = ""
        self.total_content = ""
        self.language = language
```

在init方法中我们声明了角色名称，角色类型，角色人物目的，以及constraints则是我们期望对输出内容的约束，我们希望内容最终以markdown格式输出方便我们导入到网页或者其他内容中

这里需要注意的是我们在这里只初始化了WriteDirectory动作而没有初始化WriteContent动作，为什么呢？

首先我们根据role基类中定义的_init_actions方法来看，当我们初始化一个动作时，这个动作将被加入到self._actions 中，而self._actions为一个列表，其中存储了我们所有的动作。

```python
def _init_actions(self, actions):
    self._reset()
    for idx, action in enumerate(actions):
        if not isinstance(action, Action):
            i = action("", llm=self._llm)
        else:
            if self._setting.is_human and not isinstance(action.llm, HumanProvider):
                logger.warning(f"is_human attribute does not take effect,"
                    f"as Role's {str(action)} was initialized using LLM, try passing in Action classes instead of initialized instances")
            i = action
        i.set_prefix(self._get_prefix(), self.profile)
        self._actions.append(i)
        self._states.append(f"{idx}. {action}")
        # 最后输出的样例 ['0. WriteContent', '1. WriteContent', '2. WriteContent', '3. WriteContent', '4. WriteContent', '5. WriteContent', '6. WriteContent', '7. WriteContent', '8. WriteContent']
```

接着我们来查看Role基类中run方法的实现，当我们启动一个角色使他run时他会如何工作

```python
async def run(self, message=None):
    """Observe, and think and act based on the results of the observation"""
    if message:
        if isinstance(message, str):
            message = Message(message)
        if isinstance(message, Message):
            self.recv(message)
        if isinstance(message, list):
            self.recv(Message("\n".join(message)))
    elif not await self._observe():
        # If there is no new information, suspend and wait
        logger.debug(f"{self._setting}: no news. waiting.")
        return

    rsp = await self.react()
    # Publish the reply to the environment, waiting for the next subscriber to process
    self._publish_message(rsp)
    return rsp
```

首先它将接受用户的输入（message），然后观察环境信息（目前我们还不涉及这部分内容），接着我们将调用react方法来获取输出

```python
async def react(self) -> Message:
    """Entry to one of three strategies by which Role reacts to the observed Message"""
    if self._rc.react_mode == RoleReactMode.REACT:
        rsp = await self._react()
    elif self._rc.react_mode == RoleReactMode.BY_ORDER:
        rsp = await self._act_by_order()
    elif self._rc.react_mode == RoleReactMode.PLAN_AND_ACT:
        rsp = await self._plan_and_act()
    self._set_state(state=-1) # current reaction is complete, reset state to -1 and todo back to None
    return rsp
```

当我们不指定reactmode 时将会执行self._react()方法，同时执行self._set_state()方法将初始化此时状态为-1

这里的state就代表当前agent需要执行动作的下标，当state为-1时，此时没有需要执行的actiol  self._rc.todo 此时就为空

```python
def _set_state(self, state: int):
    """Update the current state."""
    self._rc.state = state
    logger.debug(self._actions)
    self._rc.todo = self._actions[self._rc.state] if state >= 0 else None
```

再来看self._react()方法

```python
async def _react(self) -> Message:
        """Think first, then act, until the Role _think it is time to stop and requires no more todo.
        This is the standard think-act loop in the ReAct paper, which alternates thinking and acting in task solving, i.e. _think -> _act -> _think -> _act -> ... 
        Use llm to select actions in _think dynamically
        """
        actions_taken = 0
        rsp = Message("No actions taken yet") # will be overwritten after Role _act
        while actions_taken < self._rc.max_react_loop:
            # think
            await self._think()
            if self._rc.todo is None:
                break
            # act
            logger.debug(f"{self._setting}: {self._rc.state=}, will do {self._rc.todo}")
            rsp = await self._act()
            actions_taken += 1
        return rsp # return output from the last action
```

self._react()方法基本决定了agent的行动路线，这里需要思考的是要实现我们期望的agent，他应该怎样行动？

我们重写_react方法如下：

我们让agent先执行self._think()方法，在一个循环中思考目前需要做什么，思考完成后执行我们的动作，当没有需要采取的行动后我们就退出循环，把最后我们得到的最终结果写入至本地

```python
async def _react(self) -> Message:
    """Execute the assistant's think and actions.

    Returns:
        A message containing the final result of the assistant's actions.
    """
    while True:
        await self._think()
        if self._rc.todo is None:
            break
        msg = await self._act()
    root_path = TUTORIAL_PATH / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    await File.write(root_path, f"{self.main_title}.md", self.total_content.encode('utf-8'))
    return msg
```

_think方法负责更新当前需要触发的行为

我们重写_think方法如下：

当目前没有需要执行的动作，也就是之前分配的动作执行结束后 self._rc.todo为None时，让他执行目前action列表中初始的action，如果当前还没有执行到目前action列表的末尾，那么就执行下一个动作，否则将目前的self._rc.todo 置为None

```python
async def _think(self) -> None:
    """Determine the next action to be taken by the role."""
    if self._rc.todo is None:
        self._set_state(0)
        return

    if self._rc.state + 1 < len(self._states):
        self._set_state(self._rc.state + 1)
    else:
        self._rc.todo = None
```

思考结束后，这个角色就该行动起来了

我们重写_act方法如下：

_act 方法中我们将目前的todo内容按照action的类型分开处理，当目前需要生成目录时我们就获取用户的输入，传入 WriteDirectory action 内 生成对应的目录，最后，在 _handle_directory 方法中根据目录内容，我们生成子任务，也就是根据标题题目来生成内容，子任务生成结束后，我们使用self._init_actions更新目前的任务列表

当下次运行_act方法时，我们就将执行WriteContent 动作，来生成指定目录中的内容

```python
async def _act(self) -> Message:
    """Perform an action as determined by the role.

    Returns:
            A message containing the result of the action.
    """
    todo = self._rc.todo
    if type(todo) is WriteDirectory:
        msg = self._rc.memory.get(k=1)[0]
        self.topic = msg.content
        resp = await todo.run(topic=self.topic)
        logger.info(resp)
        return await self._handle_directory(resp)
    resp = await todo.run(topic=self.topic)
    logger.info(resp)
    if self.total_content != "":
        self.total_content += "\n\n\n"
    self.total_content += resp
    return Message(content=resp, role=self.profile)
async def _handle_directory(self, titles: Dict) -> Message:
    """Handle the directories for the tutorial document.

    Args:
        titles: A dictionary containing the titles and directory structure,
                such as {"title": "xxx", "directory": [{"dir 1": ["sub dir 1", "sub dir 2"]}]}

    Returns:
        A message containing information about the directory.
    """
    # 当生成目录后记录目录标题（因为最后要输出完整文档）
    self.main_title = titles.get("title")
    directory = f"{self.main_title}\n"
    # self.total_content用来存储最好要输出的所有内容
    self.total_content += f"# {self.main_title}"
    actions = list()
    for first_dir in titles.get("directory"):
        # 根据目录结构来生成新的需要行动的action（目前只设计了两级目录）
        actions.append(WriteContent(language=self.language, directory=first_dir))
        key = list(first_dir.keys())[0]
        directory += f"- {key}\n"
        for second_dir in first_dir[key]:
            directory += f"  - {second_dir}\n"
    self._init_actions(actions)
    self._rc.todo = None
    return Message(content=directory)
```

如果你还没有理解，这里我制作了一个简单的思维导图来帮助你梳理这个过程

[![piRheH0.png](https://z1.ax1x.com/2023/12/10/piRheH0.png)](https://imgse.com/i/piRheH0)

### 1.4.5 运行 TutorialAssistant 角色

接下来你只需要初始化它并使用一个起始消息运行它。

```python
import asyncio

async def main():
    msg = "Git 教程"
    role = TutorialAssistant()
    logger.info(msg)
    result = await role.run(msg)
    logger.info(result)

asyncio.run(main())
```

完整代码如下：

```python
from datetime import datetime
from typing import Dict
import asyncio
from metagpt.actions.write_tutorial import WriteDirectory, WriteContent
from metagpt.const import TUTORIAL_PATH
from metagpt.logs import logger
from metagpt.roles import Role
from metagpt.schema import Message
from metagpt.utils.file import File
import fire

from typing import Dict

from metagpt.actions import Action
from metagpt.prompts.tutorial_assistant import DIRECTORY_PROMPT, CONTENT_PROMPT
from metagpt.utils.common import OutputParser

class WriteDirectory(Action):
    """Action class for writing tutorial directories.

    Args:
        name: The name of the action.
        language: The language to output, default is "Chinese".
    """

    def __init__(self, name: str = "", language: str = "Chinese", *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self.language = language

    async def run(self, topic: str, *args, **kwargs) -> Dict:
        """Execute the action to generate a tutorial directory according to the topic.

        Args:
            topic: The tutorial topic.

        Returns:
            the tutorial directory information, including {"title": "xxx", "directory": [{"dir 1": ["sub dir 1", "sub dir 2"]}]}.
        """
        COMMON_PROMPT = """
        You are now a seasoned technical professional in the field of the internet. 
        We need you to write a technical tutorial with the topic "{topic}".
        """

        DIRECTORY_PROMPT = COMMON_PROMPT + """
        Please provide the specific table of contents for this tutorial, strictly following the following requirements:
        1. The output must be strictly in the specified language, {language}.
        2. Answer strictly in the dictionary format like {{"title": "xxx", "directory": [{{"dir 1": ["sub dir 1", "sub dir 2"]}}, {{"dir 2": ["sub dir 3", "sub dir 4"]}}]}}.
        3. The directory should be as specific and sufficient as possible, with a primary and secondary directory.The secondary directory is in the array.
        4. Do not have extra spaces or line breaks.
        5. Each directory title has practical significance.
        """
        prompt = DIRECTORY_PROMPT.format(topic=topic, language=self.language)
        resp = await self._aask(prompt=prompt)
        return OutputParser.extract_struct(resp, dict)

class WriteContent(Action):
    """Action class for writing tutorial content.

    Args:
        name: The name of the action.
        directory: The content to write.
        language: The language to output, default is "Chinese".
    """

    def __init__(self, name: str = "", directory: str = "", language: str = "Chinese", *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self.language = language
        self.directory = directory

    async def run(self, topic: str, *args, **kwargs) -> str:
        """Execute the action to write document content according to the directory and topic.

        Args:
            topic: The tutorial topic.

        Returns:
            The written tutorial content.
        """
        COMMON_PROMPT = """
        You are now a seasoned technical professional in the field of the internet. 
        We need you to write a technical tutorial with the topic "{topic}".
        """
        CONTENT_PROMPT = COMMON_PROMPT + """
        Now I will give you the module directory titles for the topic. 
        Please output the detailed principle content of this title in detail. 
        If there are code examples, please provide them according to standard code specifications. 
        Without a code example, it is not necessary.

        The module directory titles for the topic is as follows:
        {directory}

        Strictly limit output according to the following requirements:
        1. Follow the Markdown syntax format for layout.
        2. If there are code examples, they must follow standard syntax specifications, have document annotations, and be displayed in code blocks.
        3. The output must be strictly in the specified language, {language}.
        4. Do not have redundant output, including concluding remarks.
        5. Strict requirement not to output the topic "{topic}".
        """
        prompt = CONTENT_PROMPT.format(
            topic=topic, language=self.language, directory=self.directory)
        return await self._aask(prompt=prompt)

class TutorialAssistant(Role):
    """Tutorial assistant, input one sentence to generate a tutorial document in markup format.

    Args:
        name: The name of the role.
        profile: The role profile description.
        goal: The goal of the role.
        constraints: Constraints or requirements for the role.
        language: The language in which the tutorial documents will be generated.
    """

    def __init__(
        self,
        name: str = "Stitch",
        profile: str = "Tutorial Assistant",
        goal: str = "Generate tutorial documents",
        constraints: str = "Strictly follow Markdown's syntax, with neat and standardized layout",
        language: str = "Chinese",
    ):
        super().__init__(name, profile, goal, constraints)
        self._init_actions([WriteDirectory(language=language)])
        self.topic = ""
        self.main_title = ""
        self.total_content = ""
        self.language = language

    async def _think(self) -> None:
        """Determine the next action to be taken by the role."""
        logger.info(self._rc.state)
        logger.info(self,)
        if self._rc.todo is None:
            self._set_state(0)
            return

        if self._rc.state + 1 < len(self._states):
            self._set_state(self._rc.state + 1)
        else:
            self._rc.todo = None

    async def _handle_directory(self, titles: Dict) -> Message:
        """Handle the directories for the tutorial document.

        Args:
            titles: A dictionary containing the titles and directory structure,
                    such as {"title": "xxx", "directory": [{"dir 1": ["sub dir 1", "sub dir 2"]}]}

        Returns:
            A message containing information about the directory.
        """
        self.main_title = titles.get("title")
        directory = f"{self.main_title}\n"
        self.total_content += f"# {self.main_title}"
        actions = list()
        for first_dir in titles.get("directory"):
            actions.append(WriteContent(
                language=self.language, directory=first_dir))
            key = list(first_dir.keys())[0]
            directory += f"- {key}\n"
            for second_dir in first_dir[key]:
                directory += f"  - {second_dir}\n"
        self._init_actions(actions)
        self._rc.todo = None
        return Message(content=directory)

    async def _act(self) -> Message:
        """Perform an action as determined by the role.

        Returns:
            A message containing the result of the action.
        """
        todo = self._rc.todo
        if type(todo) is WriteDirectory:
            msg = self._rc.memory.get(k=1)[0]
            self.topic = msg.content
            resp = await todo.run(topic=self.topic)
            logger.info(resp)
            return await self._handle_directory(resp)
        resp = await todo.run(topic=self.topic)
        logger.info(resp)
        if self.total_content != "":
            self.total_content += "\n\n\n"
        self.total_content += resp
        return Message(content=resp, role=self.profile)

    async def _react(self) -> Message:
        """Execute the assistant's think and actions.

        Returns:
            A message containing the final result of the assistant's actions.
        """
        while True:
            await self._think()
            if self._rc.todo is None:
                break
            msg = await self._act()
        root_path = TUTORIAL_PATH / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        await File.write(root_path, f"{self.main_title}.md", self.total_content.encode('utf-8'))
        return msg

async def main():
    msg = "Git 教程"
    role = TutorialAssistant()
    logger.info(msg)
    result = await role.run(msg)
    logger.info(result)

asyncio.run(main())
```

## 2. 课程任务：
经过上面的学习，我想你已经对 MetaGPT 的框架有了基本了解，现在我希望你能够自己编写这样一个 agent
- 这个 Agent 拥有三个动作 打印1 打印2 打印3
- 重写有关方法（请不要使用act_by_order，我希望你能独立实现）使得 Agent 顺序执行上面三个动作
- 当上述三个动作执行完毕后，为 Agent 生成新的动作 打印4 打印5 打印6 并顺序执行
如果完成上面的任务，那这次作业已经可以算完成了，但关于这个 Agent 我们还有更多可以思考的地方
- 目前为止我们设计的所有思考模式都可以总结为是链式的思考（chain of thought），能否利用 MetaGPT 框架实现树结构的思考（tree of thought）图结构的思考（graph of thought）？试着实现让 ai 生成树结构的动作列表，并按照树的遍历方式执行他们，如果你实现，这将是加分项


如果您对更多内容感兴趣，欢迎联系我