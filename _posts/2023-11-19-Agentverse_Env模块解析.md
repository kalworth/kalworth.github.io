---
title : Agentverse_Env模块解析
tags : python,LLM,Agent
---  
# Agentverse 介绍
AgentVerse 提供了一个多功能的框架，简化了为大型语言模型（LLMs）创建自定义多智能体环境的过程。旨在快速、低成本的开发和定制，我们的框架赋能研究人员专注于他们的研究，而不被实现细节所困扰。  
[paper](https://arxiv.org/abs/2308.10848)
## AgentVerse Env 组件设计
AgentVerse 的核心在于 环境组件的设计，他们为环境定义了如下的基础组件：  

- **Describer（描述器）**：此组件为每个智能体在每一轮提供环境的描述。您可以自定义描述器来定义他们的环境的具体要求，例如一个智能体可以与哪些智能体互动。
- **Order（顺序）**：此组件定义智能体在环境中采取行动的顺序。您可以自定义顺序以反映智能体之间所需的交互。我们提供了几个基本的顺序选项，包括`random`（随机），`sequential`（连续）和`concurrent`（所有智能体在每轮都采取行动）。
- **Selector（选择器）**：此组件选择由智能体生成的有效消息。有时智能体可能生成无效的响应，选择器用于过滤出意外的结果。
- **Updater（更新器）**：此组件更新每个智能体的记忆。在某些情况下，一个智能体生成的响应不应被所有智能体看到（例如，如果智能体在不同的房间里）。对于每个响应，更新器只更新可以看到它的智能体。
- **Visibility（可见性）**：此组件维护每个智能体在环境变化中可以看到的智能体列表。例如，当一个智能体从一个房间移动到另一个房间时，每个智能体的可见智能体列表应由`visibility`更新。  

## Env base的定义
如下：
```python
if TYPE_CHECKING:
    from agentverse.agents.base import BaseAgent
    from agentverse.message import Message


class BaseRule(BaseModel):
    pass


class BaseEnvironment(BaseModel):
    """
    Base class for environment.

    Args:
        agents: List of agents
        rule: Rule for the environment
        max_turns: Maximum number of turns
        cnt_turn: Current turn number
        last_messages: Messages from last turn
        rule_params: Variables set by the rule
    """

    agents: List[BaseAgent]
    rule: BaseRule
    max_turns: int = 10
    cnt_turn: int = 0
    last_messages: List[Message] = []
    rule_params: Dict = {}

    @abstractmethod
    async def step(self) -> List[Message]:
        """Run one step of the environment"""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset the environment"""
        pass

    def report_metrics(self) -> None:
        """Report useful metrics"""
        total_spent = sum([agent.get_spend() for agent in self.agents])
        logger.info(f"Total spent: ${total_spent}")
        pass

    def is_done(self) -> bool:
        """Check if the environment is done"""
        return self.cnt_turn >= self.max_turns
```
Env 中包含：  
* agents: List of agents
* **rule: Rule for the environment** 核心，其中定义了agent 与环境交互的基本规则
* max_turns: Maximum number of turns
* cnt_turn: Current turn number
* last_messages: Messages from last turn
* rule_params: Variables set by the rule

## Agentverse 提供的Env样例


- simulation：BaseEnv
- ReflectionEnvironment
- sPokemonEnvironment
- PrisonerDilemmaEnvironment
- SdeTeamEnvironment
- SdeTeamGivenTestsEnvironment
- BasicEnvironment

tasks : simulation , tasksolving
## BaseEnv解析

Agentverse 将Env 定义为两大类，分别为 simulation（模拟） tasksolving（任务解决）  
 
### tasksolving
```python
class BasicEnvironment(BaseEnvironment):

    # 导入Tasksolving_rule 
    # Tasksolving内定义了如下规则：role_assigner（角色分配）decision_maker（决策制订）executor（执行）evaluator（评估）
    rule: TasksolvingRule
    agents: Dict[Enum, Union[BaseAgent, List[BaseAgent]]] = None

    task_description: str

    cnt_turn: int = 0
    max_turn: int = 10
    success: bool = False

    # 初始化环境
    def __init__(self, **kwargs):
        rule_config = kwargs.pop("rule", {})
        role_assigner_config = rule_config.pop(
            "role_assigner", {"type": "role_description"}
        )
        decision_maker_config = rule_config.pop("decision_maker", {"type": "vertical"})
        executor_config = rule_config.pop("executor", {"type": "none"})
        evaluator_config = rule_config.pop("evaluator", {"type": "basic"})
        rule = TasksolvingRule(
            role_assigner_config=role_assigner_config,
            decision_maker_config=decision_maker_config,
            executor_config=executor_config,
            evaluator_config=evaluator_config,
        )
        super().__init__(rule=rule, **kwargs)
    # step 方法用于执行
    async def step(
        self, advice: str = "No advice yet.", previous_plan: str = "No solution yet."
    ) -> List[Message]:
        result = ""
        logs = []
        # 记录当前回合
        logger.info(f"Loop Round {self.cnt_turn}")

        # ================== EXPERT RECRUITMENT ==================
        # 分配角色，根据（任务描述）task_description （智能体）agents （目前回合）cnt_turn （建议）advice 分配
        agents = await self.rule.role_assign(
            self.task_description, self.agents, self.cnt_turn, advice
        )
        description = "\n".join([agent.role_description for agent in agents])
        logs.append({"module": "Role Assigner", "content": description})
        logger.info("", f"Role Assignment:\n{description}", Fore.CYAN)
        # ================== EXPERT RECRUITMENT ==================

        # ================== DECISION MAKING ==================
        # 调用decision_maker 中的decision_making方法返回生成的计划
        plan: List[SolverMessage] = await self.rule.decision_making(
            self.task_description, self.agents, previous_plan, advice
        )
        flatten_plan = "\n".join([p.content for p in plan])
        logs.append({"module": "Decision Maker", "content": flatten_plan})
        logger.info("", f"Decision Plan:\n{flatten_plan}", Fore.YELLOW)
        # ================== DECISION MAKING ==================

        # ================== EXECUTION ==================
        # 按照plan执行一轮动作
        result: List[ExecutorMessage] = await self.rule.execute(
            self.task_description, self.agents, plan
        )
        flatten_result = "\n".join([r.content for r in result])
        logs.append({"module": "Executor", "content": flatten_result})
        logger.info("", f"Execution Result:", Fore.GREEN)
        logger.info("", flatten_result, Fore.GREEN)
        # ================== EXECUTION ==================

        # ================== EVALUATION ==================
        # 执行结束后由evaluate来生成对任务的评价与后续优化的建议
        score, advice = await self.rule.evaluate(
            self.task_description, self.agents, plan, result
        )
        logs.append(
            {
                "agent": "evaluator",
                "content": f"Evaluation result: Score: {score}\nAdvice: {advice}",
            }
        )
        logger.info(
            "", f"Evaluation result:\nScore: {score}\nAdvice: {advice}", Fore.YELLOW
        )

        if score is not None and (
            # 单任务结果为完成
            (isinstance(score, bool) and score is True)
            # 多任务打分均在 8 以上 为完成
            or (isinstance(score, (list, tuple)) and all([s >= 8 for s in score]))
        ):
            # TODO: 8 is an arbitrary threshold
            logs.append({"agent": "system", "content": "Good score! Accept!"})
            logger.info(
                "", f"Good score! Accept! Final Result:\n{flatten_plan}", Fore.GREEN
            )
            self.success = True
        else:
            logs.append({"agent": "system", "content": "Bad score! Reject!"})
            logger.info("", "Bad score! Reject!", Fore.RED)
        self.cnt_turn += 1
        return flatten_result, advice, flatten_plan, logs, self.success
    # 遍历 agents
    def iter_agents(self):
        for role, agent_or_agents in self.agents.items():
            if isinstance(agent_or_agents, list):
                for agent in agent_or_agents:
                    yield role, agent
            else:
                yield role, agent_or_agents
    # 获取每个agent 的token花费
    def get_spend(self):
        total_spent = sum([agent.get_spend() for (_, agent) in self.iter_agents()])
        return total_spent
    # report token 花费
    def report_metrics(self) -> None:
        logger.info("", "Agent spend:", Fore.GREEN)
        for role, agent in self.iter_agents():
            name = agent.name.split(":")[0]
            logger.info(
                "",
                f"Agent (Role: {role}) {name}: {agent.get_spend_formatted()}",
                Fore.GREEN,
            )
        logger.info("", f"Total spent: ${self.get_spend():.6f}", Fore.GREEN)

    def is_done(self):
        """Check if the environment is done"""
        return self.cnt_turn >= self.max_turn or self.success

    def set_task_description(self, task_description: str = ""):
        self.task_description = task_description

    # 重置环境
    def reset(self) -> None:
        """Reset the environment"""
        self.cnt_turn = 0
        self.rule.reset()
```
上面的内容仅阐述了 Env 的基本定义，接下来我们来看 TasksolvingRule 的内容  

```python
class TasksolvingRule(BaseRule):
    # role_assigner（角色分配）decision_maker（决策制订）executor（执行）evaluator（评估）
    role_assigner: BaseRoleAssigner
    decision_maker: BaseDecisionMaker
    executor: BaseExecutor
    evaluator: BaseEvaluator


    role_assign_only_once: bool = False
    add_execution_result_to_critic: bool = False
    add_execution_result_to_solver: bool = False

    def __init__(
        self,
        role_assigner_config,
        decision_maker_config,
        executor_config,
        evaluator_config,
        *args,
        **kwargs,
    ):
        # 根据yaml 配置文件内内容初始化组件
        def build_components(config: Dict, registry):
            component_type = config.pop("type")
            component = registry.build(component_type, **config)
            return component

        role_assigner = build_components(
            role_assigner_config,
            role_assigner_registry,
        )
        decision_maker = build_components(
            decision_maker_config,
            decision_maker_registry,
        )
        executor = build_components(executor_config, executor_registry)
        evaluator = build_components(evaluator_config, evaluator_registry)
        super().__init__(
            role_assigner=role_assigner,
            decision_maker=decision_maker,
            executor=executor,
            evaluator=evaluator,
            *args,
            **kwargs,
        )

    # 角色分配，初始化agent
    async def role_assign(
        self,
        task_description: str,
        agents: List[BaseAgent],
        cnt_turn: int,
        advice: str = "",
    ) -> List[BaseAgent]:
        """Assign roles to agents"""
        if self.role_assign_only_once and cnt_turn > 0:
            agents = [agents[AGENT_TYPES.SOLVER]] + agents[AGENT_TYPES.CRITIC]
        else:
            agents = await self.role_assigner.astep(
                role_assigner=agents[AGENT_TYPES.ROLE_ASSIGNMENT],
                group_members=[agents[AGENT_TYPES.SOLVER]] + agents[AGENT_TYPES.CRITIC],
                advice=advice,
                task_description=task_description,
            )
            if self.role_assign_only_once and cnt_turn == 0:
                agents[AGENT_TYPES.SOLVER] = agents[0]
                agents[AGENT_TYPES.CRITIC] = agents[1:]
        return agents

    # 生成决策
    async def decision_making(
        self,
        task_description: str,
        agents: List[BaseAgent],
        previous_plan: str,
        advice: str = "No advice yet.",
    ) -> List[SolverMessage]:
        # TODO: plan should be string or a special type of object?

        # dynamic
        if "dynamic" in self.decision_maker.name:
            plan = await self.decision_maker.astep(
                agents=[agents[AGENT_TYPES.SOLVER], *agents[AGENT_TYPES.CRITIC]],
                manager=agents[AGENT_TYPES.MANAGER],
                task_description=task_description,
                previous_plan=previous_plan,
                advice=advice,
            )
        else:
            plan = await self.decision_maker.astep(
                agents=[agents[AGENT_TYPES.SOLVER], *agents[AGENT_TYPES.CRITIC]],
                task_description=task_description,
                previous_plan=previous_plan,
                advice=advice,
            )
        return plan

    # 执行
    async def execute(
        self,
        task_description: str,
        agents: List[BaseAgent],
        final_solution: List[SolverMessage],
    ) -> Any:
        """execution stage.
        Use the executor to finish the task.
        """

        results = await self.executor.astep(
            agents[AGENT_TYPES.EXECUTION], task_description, final_solution
        )
        if self.add_execution_result_to_critic:
            for agent in agents[AGENT_TYPES.CRITIC]:
                agent.add_message_to_memory(results)
        if self.add_execution_result_to_solver:
            agents[AGENT_TYPES.SOLVER].add_message_to_memory(results)
        return results

    # 评估结果
    async def evaluate(
        self,
        task_description: str,
        agents: List[BaseAgent],
        solution: List[SolverMessage],
        result: List[ExecutorMessage],
    ) -> Tuple[List[int], str]:
        """evaluation stage."""
        # if self.human_eval:
        #     print("This round, LLM gave the following result:")
        #     print(result)
        #     comprehensiveness = input("Please evaluate the comprehensiveness>> ")
        #     detailedness = input("Please evaluate the detailedness>> ")
        #     feasibility = input("Please evaluate the feasibility>> ")
        #     novelty = input("Please evaluate the novelty>> ")
        #     advice = input("Please give some advice>>")
        #     try:
        #         comprehensiveness = int(comprehensiveness)
        #         detailedness = int(detailedness)
        #         feasibility = int(feasibility)
        #         novelty = int(novelty)
        #     except ValueError:
        #         logger.error("Bad response from human evaluator!")
        #     return ([comprehensiveness, detailedness, feasibility, novelty], advice)
        # else:
        evaluation = await self.evaluator.astep(
            agent=agents[AGENT_TYPES.EVALUATION],
            solution=solution,
            result=result,
            task_description=task_description,
            all_role_description=[
                agents[AGENT_TYPES.SOLVER].role_description,
                *[agent.role_description for agent in agents[AGENT_TYPES.CRITIC]],
            ],
        )
        # 返回任务得分，返回评估后的修改建议
        return evaluation.score, evaluation.advice

    def reset(self) -> None:
        self.role_assigner.reset()
        self.decision_maker.reset()
        self.executor.reset()
        self.evaluator.reset()

```
env rule 中规定了角色分配的方式 ，如何决策以及执行决策和评估角色决策过程，agentverse中，决策，执行决策，评估决策都通过分配Agent的角色来完成

### simulation
agentverse 中把不需要明确目标的任务称为 simulation（模拟）
```python
@EnvironmentRegistry.register("sim-basic")
class BasicEnvironment(BaseEnvironment):
    """
    A basic environment implementing the logic of conversation.

    Args:
        agents: List of agents
        rule: Rule for the environment
        max_turns: Maximum number of turns
        cnt_turn: Current turn number
        last_messages: Messages from last turn
        rule_params: Variables set by the rule
    """

    agents: List[BaseAgent]
    rule: Rule
    max_turns: int = 10
    cnt_turn: int = 0
    last_messages: List[Message] = []
    rule_params: Dict = {}
    # 初始化环境
    def __init__(self, rule, **kwargs):
        rule_config = rule
        order_config = rule_config.get("order", {"type": "sequential"})
        visibility_config = rule_config.get("visibility", {"type": "all"})
        selector_config = rule_config.get("selector", {"type": "basic"})
        updater_config = rule_config.get("updater", {"type": "basic"})
        describer_config = rule_config.get("describer", {"type": "basic"})
        rule = Rule(
            order_config,
            visibility_config,
            selector_config,
            updater_config,
            describer_config,
        )
        super().__init__(rule=rule, **kwargs)
    # 按照规定的agent 顺序执行agent
    async def step(self) -> List[Message]:
        """Run one step of the environment"""

        # Get the next agent index
        agent_ids = self.rule.get_next_agent_idx(self)

        # Generate current environment description
        # 根据不同的agent 生成对应的 env description
        env_descriptions = self.rule.get_env_description(self)
        
        # Generate the next message
        # agent 根据自己对应的描述采取行动
        messages = await asyncio.gather(
            *[self.agents[i].astep(env_descriptions[i]) for i in agent_ids]
        )

        # Some rules will select certain messages from all the messages
        selected_messages = self.rule.select_message(self, messages)
        self.last_messages = selected_messages
        self.print_messages(selected_messages)

        # Update the memory of the agents
        self.rule.update_memory(self)

        # Update the set of visible agents for each agent
        self.rule.update_visible_agents(self)

        self.cnt_turn += 1

        return selected_messages

    def print_messages(self, messages: List[Message]) -> None:
        for message in messages:
            if message is not None:
                # logging.info(f"{message.sender}: {message.content}")
                logger.info(f"{message.sender}: {message.content}")

    def reset(self) -> None:
        """Reset the environment"""
        self.cnt_turn = 0
        self.rule.reset()
        for agent in self.agents:
            agent.reset()

    def is_done(self) -> bool:
        """Check if the environment is done"""
        return self.cnt_turn >= self.max_turns

```
下面介绍  SimulationRule   
其中使用了 agentverse核心的env 组件 也是agentverse最大的亮点
```python
class SimulationRule(BaseRule):
    """
    Rule for the environment. It controls the speaking order of the agents
    and maintain the set of visible agents for each agent.
    """

    order: BaseOrder
    visibility: BaseVisibility
    selector: BaseSelector
    updater: BaseUpdater
    describer: BaseDescriber

    def __init__(
        self,
        order_config,
        visibility_config,
        selector_config,
        updater_config,
        describer_config,
    ):
        order = order_registry.build(**order_config)
        visibility = visibility_registry.build(**visibility_config)
        selector = selector_registry.build(**selector_config)
        updater = updater_registry.build(**updater_config)
        describer = describer_registry.build(**describer_config)
        super().__init__(
            order=order,
            visibility=visibility,
            selector=selector,
            updater=updater,
            describer=describer,
        )
    # 根据order 获取agent id，按顺序执行
    '''
    class BaseOrder(BaseModel):
    @abstractmethod
    def get_next_agent_idx(self, environment: BaseEnvironment) -> List[int]:
        """Return the index of the next agent to speak"""

    def reset(self) -> None:
        pass
    '''
    def get_next_agent_idx(
        self, environment: BaseEnvironment, *args, **kwargs
    ) -> List[int]:
        """Return the index of the next agent to speak"""
        return self.order.get_next_agent_idx(environment, *args, **kwargs)
    # 
    '''
    class BaseVisibility(BaseModel):
    @abstractmethod
    def update_visible_agents(self, environment: BaseEnvironment):
        """Update the set of visible agents for the agent"""

    def reset(self):
        pass

    '''
    def update_visible_agents(
        self, environment: BaseEnvironment, *args, **kwargs
    ) -> None:
        """Update the set of visible agents for the agent"""
        self.visibility.update_visible_agents(environment, *args, **kwargs)

    # 利用selecor 过滤agent回复的无效信息
    '''
    class BaseSelector(BaseModel):
    @abstractmethod
    def select_message(
        self, environment: BaseEnvironment, messages: List[Message]
    ) -> List[Message]:
        """Selects a set of valid messages from all messages"""
        pass

    def reset(self) -> None:
        pass
    '''
    def select_message(
        self, environment: BaseEnvironment, messages: List[Message], *args, **kwargs
    ) -> List[Message]:
        """Select a set of valid messages from all the generated messages"""
        return self.selector.select_message(environment, messages, *args, **kwargs)

    # 根据事件发生的结果更新agent的记忆（根据visible属性，也即每个agent获取信息的权限）
    '''
    class BaseUpdater(BaseModel):
    """
    The base class of updater class.
    """

    @abstractmethod
    def update_memory(self, environment: BaseEnvironment):
        pass

    def reset(self):
        pass

    '''
    def update_memory(self, environment: BaseEnvironment, *args, **kwargs) -> None:
        """For each message, add it to the memory of the agent who is able to see that message"""
        self.updater.update_memory(environment, *args, **kwargs)

    # 生成对环境的描述
    '''
    class BaseDescriber(BaseModel):
    @abstractmethod
    def get_env_description(
        self, environment: BaseEnvironment, *args, **kwargs
    ) -> List[str]:
        """Return the environment description for each agent"""
        pass

    def reset(self) -> None:
        pass
    '''
    def get_env_description(
        self, environment: BaseEnvironment, *args, **kwargs
    ) -> List[str]:
        """Return the description of the environment for each agent"""
        return self.describer.get_env_description(environment, *args, **kwargs)

    def reset(self) -> None:
        self.order.reset()
        self.visibility.reset()
        self.selector.reset()
        self.updater.reset()
        self.describer.reset()

```
## yaml 管理env
而关于环境的更细节的配置，由更具体的程序实现，再由用户声明的配置文件（yaml导入）  
eg ：Pokemon env  
```yaml
prompts:
  prompt: &prompt |-
    Now you are in the world of Pokémon Emerald, living as one of the characters. Brendan, a key character in the Pokémon Emerald world, will interact with you during your journey. Pay close attention to his conversations and respond authentically as your character. Your choices and dialogue will shape the course of your adventure. When you give your response, you should always output in the following format:
    Thought: (your thought here)
    Action: (an action name, can be Speak, MoveTo, or other actions)
    Action Input: (the arguments for the action in json format, and NOTHING else)

    For example, when you would like to talk to person XX, you can output in the following format:
    Thought: (your thought here)
    Action: Speak
    Action Input: {"to": "XX", "text": "..."}

    When you would like to do something in the current place, you can output in the following format:
    Thought: (your thought here)
    Action: (action_name)
    Action Input: {"last_time": "xx minutes"}

    When you would like to move to another place, you can output in the following format:
    Thought: (your thought here)
    Action: MoveTo
    Action Input: {"to": "name_of_the_place"}

    The places you can go include:
    - Pokémon Center: The Pokémon Center is a place where you can get your Pokémon healed. A Pokémon Center is completely free of charge and is found in most major cities.
    - Shop: The Shop is a place where you can buy the daily necessities.
    - Bike Store: The Bike Store is a place where you can buy a bike.
    - Park: The Park is a place where you can relax yourself. Many residents in the town like to go there to chat with others.
    - Pokémon Gym: The Pokémon Gym is a place where Pokémon Trainers can battle Gym Leaders to earn Badges. These Badges serve as proof of a Trainer's skill and are necessary to enter the Pokémon League, which is a tournament where Trainers compete to become the regional Champion.
    
    ${role_description} 
    Now, immerse yourself in this vibrant world and let your character's personality shine. Good luck!

    Here is the conversation history so far:
    ${chat_history}
    ${env_description}

    What will you, ${agent_name}, do next?

environment:
  env_type: pokemon
  max_turns: 10000000
  locations:
    - name: Pokémon Center
      # description: The Pokémon Center is a place where you can get your Pokémon healed. A Pokémon Center is completely free of charge and is found in most major cities.
      init_agents: 
        - Maxie
    - name: Shop
      # description: The Shop is a place where you can buy the daily necessities.
      init_agents: 
        - Archie
    - name: Bike Store
      # description: The Bike Store is a place where you can buy a bike.
      init_agents: 
        - Joseph
    - name: Park
      # description: The Park is a place where you can relax yourself. Many residents in the town like to go there to chat with others.
      init_agents: 
        - May
        - Birch
    - name: Pokémon Gym
      # description: The Pokémon Gym is a place where Pokémon Trainers can battle Gym Leaders to earn Badges. These Badges serve as proof of a Trainer's skill and are necessary to enter the Pokémon League, which is a tournament where Trainers compete to become the regional Champion.
      init_agents: 
        - Steven
  rule:
    order:
      type: sequential
    visibility:
      type: pokemon
    selector:
      type: pokemon
    updater:
      type: pokemon
    describer:
      type: pokemon

agents:
  - agent_type: conversation
    name: May
    role_description: |-
      You are May, a character in Pokémon Emerald. You are helping your dad, Professor Birch, finish the Hoenn Pokédex and becoming a Pokémon Professor. You are also Brendan's rival and friend. For a reference, here are some quotes from you:
      "There isn't a single Trainer left in Hoenn who doesn't know who you are, Brendan! When I tell people that I'm friends with you, Brendan, they're all surprised!"
      "I wonder where I should go catch some Pokémon next? Wouldn't it be funny if we ran into each other, Brendan?"npcs.yaml
      "I'm thinking of going back to Littleroot soon. I've caught a decent group of Pokémon, and my Pokédex is coming along, so I'm going home to show my dad. Brendan, what are you going to do? Collect all the Gym Badges and take the Pokémon League challenge? Well, while you're collecting Badges, Brendan, I'm going to work on my Pokédex. I'll complete it before you! See you!"
    memory:
      memory_type: chat_history
    prompt_template: *prompt
    output_parser:
      type: pokemon
    llm:
      llm_type: gpt-3.5-turbo
      model: 'gpt-3.5-turbo'
      temperature: 0.7
      max_tokens: 1024
      stop: |+

        
  - agent_type: conversation
    name: Birch
    role_description: |-
      You are Professor Birch, a character in Pokémon Emerald. You are the resident Pokémon Professor of Littleroot Town and the Hoenn region. You specializes in Pokémon habitats and distribution. You are the father of May. You often works with your child to help observe and capture wild Pokémon. Your wife worries about you, because you are always busy and rarely has time to come home. You are known to be more outgoing than the other Pokémon Professors, and oftentimes your research takes you outdoors. Your field of study is primarily how Pokémon behave in the wild. For a reference, here are some quotes from you:
      "Oh, hi, Brendan! I heard you beat May on your first try. That's excellent! May's been helping with my research for a long time. May has an extensive history as a Trainer already. Here, Brendan, I ordered this for my research, but I think you should have this Pokédex."
      "See? What did I tell you, May? Didn't I tell you that you don't need to worry about Brendan? ... Brendan, you've finally done it. When I heard that you defeated your own father at the Petalburg Gym, I thought perhaps you had a chance... But to think you've actually become the Champion! Ah, yes! What become of your Pokédex? Here, let me see."
      "Well, well, Brendan! That was good work out there! I knew there was something special about you when I first saw you, but I never expected this. Oh, yes. Do you still have the Pokédex I gave you? I have something to show you. Let's go to my Lab."
    memory:
      memory_type: chat_history
    prompt_template: *prompt
    output_parser:
      type: pokemon
    llm:
      llm_type: gpt-3.5-turbo
      model: gpt-3.5-turbo
      temperature: 0.7
      max_tokens: 1024
      stop: |+

        
  - agent_type: conversation
    name: Steven
    role_description: |-
      You are Steven Stone, a character in Pokémon Emerald. You are the son of Joseph Stone, who is the president of Devon Corporation. You are a skilled Trainer who specializes in Steel-type Pokémon. You are the Champion of the Hoenn region's Pokémon League. You are a collector of rare stones, and you are the son of the president of the Devon Corporation, and you make your home in Mossdeep City. You wanders the region, aiding the player on their journey. You are just defeated by Brendan. For a reference, here are some quotes from you:
      "Your Pokémon appear quite capable. If you keep training, you could even become the Champion of the Pokémon League one day. That's what I think. I know, since we've gotten to know each other, let's register one another in our PokéNavs. ... Now, I've got to hurry along."
      "I see... Your battle style is intriguing. Your Pokémon have obviously grown since I first met you in Dewford. I'd like you to have this Devon Scope. Who knows, there may be other concealed Pokémon. Brendon, I enjoy seeing Pokémon and Trainers who strive together. I think you're doing great. Well, let's meet again somewhere."
      "Hi, Brendon! When you're on an adventure with your Pokémon, what do you think? Do you consider them to be strong partners? Do you think of them as fun companions? Depending on how you think, your adventure's significance changes."
    memory:
      memory_type: chat_history
    prompt_template: *prompt
    output_parser:
      type: pokemon
    llm:
      llm_type: gpt-3.5-turbo
      model: gpt-3.5-turbo
      temperature: 0.7
      max_tokens: 1024
      stop: |+

        
  - agent_type: conversation
    name: Maxie
    role_description: |-
      You are Maxie, a character in Pokémon Emerald. You are the head of Team Magma. You are the leader of Team Magma. You pursue the ideal world for humanity. You are neurotic and easily gets worked up over trivial matters, often using numbers to express various things. You possess a calm and composed personality, you also exhibit a ruthless and merciless side towards anything that obstructs you. Your ambition is to use the legendary Pokémon Groudon's power to dry up the sea and expand the land, increasing the space for terrestrial creatures to thrive. For a reference, here are some quotes from you
      "Now you listen. Long ago, living things used the land to live and grow. That is why land is all important! It is the cradle of all! That is why Team Magma is dedicated to the expansion of the land mass. It is for further advancement of humankind and Pokémon! And for that, we need the power of what sleeps within this mountain..."
      "Clear out of the way! Don't you dare interfere!"
      "Fufufu... Since you're so curious, you deserve an explanation. We're going to jettison the entire load into Mt. Chimney! With Groudon gone, we have no need for that slag heap of a mountain! So we'll use the fuel's power to make the volcano erupt! It will be savage!"
    memory:
      memory_type: chat_history
    prompt_template: *prompt
    output_parser:
      type: pokemon
    llm:
      llm_type: gpt-3.5-turbo
      model: gpt-3.5-turbo
      temperature: 0.7
      max_tokens: 1024
      stop: |+

        
  - agent_type: conversation
    name: Archie
    role_description: |-
      You are Archie, a character in Pokémon Emerald. You are the leader of Team Aqua, driven by the pursuit of establishing an ideal hometown for Pokémon. Your generous personality earns you the trust of your subordinates, and you use your strength to overcome obstacles from your opponents. However, in your pursuit of your ideals, you disregards yourself and the Team Aqua, making you a dangerous individual. For a reference, here are some quotes from you:
      "We are Team Aqua, and we love the sea! And I am Team Aqua's leader, Archie! What makes you interfere with us? ... All life depends on the sea. So, Team Aqua is dedicated to the expansion of the sea. Don't you agree? What we are doing is a magnificent undertaking. Ah, fine... You're still too young to understand our noble objective. But, I warn you, don't even consider interfering with our plans again. The consequences will cost you dearly! And don't you forget it!"
      "Brendan! Thank you! With your help, we thwarted Team Magma's destructive plan! But... You... Whose side are you on? Ah, it doesn't matter. We will remain vigilant and keep up our pursuit of Team Magma. Brendan, we shall meet again!"
      "Hold it right there. Fufufu... So it was you, after all. Behold! See how beautiful it is, the sleeping form of the ancient Pokémon Kyogre! I have waited so long for this day to come... It surprises me, how you've managed to chase me here. But that's all over now. For the realization of my dream, you must disappear now!"
    memory:
      memory_type: chat_history
    prompt_template: *prompt
    output_parser:
      type: pokemon
    llm:
      llm_type: gpt-3.5-turbo
      model: gpt-3.5-turbo
      temperature: 0.7
      max_tokens: 1024
      stop: |+

        
  - agent_type: conversation
    name: Joseph
    role_description: |-
      You are Joseph Stone, a character in Pokémon Emerald. You are the president of Devon Corporation and father of Steven Stone, who is the champion of the Hoenn region's Pokémon League. You considers yourself generous and collects rare rocks and stones. You also have a large interest in Pokémon; you wanted the PokéNav developed so that he could better understand Pokémon emotions. You prioritize quality products and is prepared to invest in experimental technology rather than profit. You also make a point to know all of your employees, namely for security purposes. You also try to escape from your business duties so you could walk the streets of Rustboro City in search of new inventions. You often speaks with children, finding their young and inquisitive minds to be among the biggest sources of inspiration. For a reference, here are some quotes from you:
      "Oho! Th-that Pokémon you have... Could it be that rare white specimen? There cannot be more than one such specimen in the world! So pure... So sublime... Its sparkle is indeed the ultimate! I would love to see how it would stand up to Steven's Beldum..."
      "Thanks to the heroic actions of you young people and your Pokémon teams... we have been able to dispel the threat of the asteroid without a single loss! Perhaps I have put too much of my faith in technology's promise alone. Perhaps my belief that the sacrifice of some might be necessary to guarantee the safety of many - Perhaps it was wrong all along."
      "Ahem! Oh, I know. I know what you want to say. My, what a hasty, impatient one you are! What are we to do with such an impatient one for our Pokémon League Champion? ...Hm? Oh, is that so? So you're the new Champion, <player>? Then I guess we'll never break you of that impatience after all, Steven! Ho ho ho ho!"
    memory:
      memory_type: chat_history
    prompt_template: *prompt
    output_parser:
      type: pokemon
    llm:
      llm_type: gpt-3.5-turbo
      model: gpt-3.5-turbo
      temperature: 0.7
      max_tokens: 1024
      stop: |+

        
tools:

```

最后做一个总结：
当 agentverse 中的示例启动时，他会读取配置文件中的内容，生成agent，根据env中的 location 描述 与 交互 order 环境describer 中生成运行环境 ，在运行中，env会按照order 给定的顺序运行对应的agent 然后按照 rule 中规定的 可见性 更新agent 的记忆 以上就是agentverse 运行的基本流程