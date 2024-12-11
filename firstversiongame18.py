import random
import numpy as np
from vispy import scene
from vispy.app import Timer
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
from gymnasium.spaces import Discrete, Box, Dict
from matplotlib.animation import FuncAnimation
from threading import Thread
import time
from scipy.spatial import KDTree
width, height = 3840, 2160  # Dimensions of the environment
vision_radius = 200  # Radius within which agents can see
vision_angle = np.pi
hearing_radius = 300  # Radius within which agents can hear
energy_threshold = 10  # Energy threshold for certain actions
berry_energy = 20  # Energy gained from consuming a berry
max_visible_agents = 5  # Maximum number of agents an agent can observe
num_species = 1  # Number of species (both predators and prey are divided into species)

# Berry reproduction parameters
berry_energy = 20  # Energy provided by a berry
berry_reproduction_radius = 50.0  # Perturbation range for new berry positions
berry_reproduction_interval = 100  # Time steps between berry reproductions

# Movement speed parameters
normal_speed = 15.0  # Normal speed of agents
high_speed_factor = 3.0  # Multiplier for high speed
high_speed_chance = 0.2  # Probability of an agent moving at high speed
position_perturbation = 5.0
velocity_perturbation = 1.0

# 通用的代理类
class Agent:
    # 全局计数器，用于生成唯一名字
    _id_counter = 0
    def __init__(self, position, velocity, energy, species,agent_type, reproduction_intention=False):
        self.position = np.array(position)# Position of the agent
        self.velocity = np.array(velocity)# Velocity of the agent
        self.energy = energy
        self.species = species  # # Species of the agent
        self.reproduction_intention = reproduction_intention# Whether the agent intends to reproduce
        self.type = agent_type  
        self.last_energy = energy  
        # 自动生成名字
        self.name = f"{agent_type}_{species}_{Agent._id_counter}"
        Agent._id_counter += 1  # 计数器自增

    def move(self):
        # Move the agent based on its velocity
        self.position += self.velocity
        if self.position[0] < 0 or self.position[0] > width:
            self.velocity[0] = -self.velocity[0]
        if self.position[1] < 0 or self.position[1] > height:
            self.velocity[1] = -self.velocity[1]

    def observe(self, agents, berries=[]):
        # 视觉信息
        vision_data = []
        for other in agents:
            if other == self:
                continue  # 跳过自身
            relative_position = other.position - self.position
            distance = np.linalg.norm(relative_position)
            
            # 判断是否在视觉范围内
            if distance <= vision_radius:
                angle = np.arctan2(relative_position[1], relative_position[0])
                if abs(angle) <= vision_angle / 2:
                    dx = relative_position[0] / vision_radius  # 归一化相对位置 x
                    dy = relative_position[1] / vision_radius  # 归一化相对位置 y
                    reproduction = int(other.reproduction_intention)  # 生育意图
                    species = other.species  # 种类
                    agent_type = 1 if isinstance(other, Predator) else 0  # 捕食者为 1，猎物为 0

                    vision_data.append([dx, dy, reproduction, species, agent_type])

        # 填充视觉数据的默认值
        while len(vision_data) < max_visible_agents:
            vision_data.append([0, 0, 0, 0, 0])
        vision_data = vision_data[:max_visible_agents]  # 限制最大可见智能体数量

        # 听觉信息
        hearing_data = []
        for other in agents:
            if other == self:
                continue  # 跳过自身
            relative_position = other.position - self.position
            distance = np.linalg.norm(relative_position)
            
            # 判断是否在听觉范围内
            if distance <= hearing_radius:
                direction = 0 if relative_position[0] < 0 else 1  # 方向（左/右）
                intensity = 1 - (distance / hearing_radius)  # 听觉强度，距离越远强度越低
                hearing_data.append([direction, intensity])

        # 填充听觉数据的默认值
        while len(hearing_data) < max_visible_agents:
            hearing_data.append([0, 0])
        hearing_data = hearing_data[:max_visible_agents]  # 限制最大可听智能体数量

        # 能量变化
        # delta_energy = self.energy - self.last_energy
        delta_energy = np.array([self.energy - self.last_energy], dtype=np.float32)
        self.last_energy = self.energy
        vision_data = np.array(vision_data[:max_visible_agents], dtype=np.float32)
        hearing_data = np.array(hearing_data[:max_visible_agents], dtype=np.float32)


            # 返回观测结果
        return {
            "observation": {
                "vision": vision_data,
                "hearing": hearing_data,
                "energy_change": delta_energy
            }
        }
    def optimized_observe(self,agent, all_agents):
        """
        优化后的观察函数，使用 KDTree 加速邻居搜索，同时保持返回格式与原函数一致。
        - agent: 当前观察的智能体
        - all_agents: 所有智能体列表
        - berries: 当前浆果列表
        """
        # 提取所有目标的位置
        all_positions = [other.position for other in all_agents if other != agent]
        all_berries_positions = [berry.position for berry in berries]
        all_positions.extend(all_berries_positions)

        # 构建 KDTree
        tree = KDTree(all_positions)

        # 查询视觉范围内的对象
        neighbors_indices = tree.query_ball_point(agent.position, r=vision_radius)

        # 视觉信息
        vision_data = []
        for idx in neighbors_indices:
            relative_position = all_positions[idx] - agent.position
            distance = np.linalg.norm(relative_position)
            angle = np.arctan2(relative_position[1], relative_position[0])
            if abs(angle) <= vision_angle / 2:
                dx = relative_position[0] / vision_radius  # 归一化 x
                dy = relative_position[1] / vision_radius  # 归一化 y

                # 判断是智能体还是浆果
                if idx < len(all_agents) - 1:  # 智能体
                    other = all_agents[idx]
                    reproduction = int(other.reproduction_intention)  # 生育意图
                    species = other.species  # 种类
                    agent_type = 1 if isinstance(other, Predator) else 0  # 捕食者为 1，猎物为 0
                else:  # 浆果
                    reproduction = 0  # 浆果没有生育意图
                    species = -1  # 表示非智能体
                    agent_type = -1  # 表示非智能体

                vision_data.append([dx, dy, reproduction, species, agent_type])

        # 填充默认值
        while len(vision_data) < max_visible_agents:
            vision_data.append([0, 0, 0, 0, 0])
        vision_data = np.array(vision_data[:max_visible_agents], dtype=np.float32)

        # 查询听觉范围内的对象
        neighbors_indices = tree.query_ball_point(agent.position, r=hearing_radius)
        hearing_data = []
        for idx in neighbors_indices:
            relative_position = all_positions[idx] - agent.position
            distance = np.linalg.norm(relative_position)
            direction = 0 if relative_position[0] < 0 else 1  # 左右方向
            intensity = 1 - (distance / hearing_radius)  # 听觉强度，距离越远强度越低
            hearing_data.append([direction, intensity])

        # 填充听觉默认值
        while len(hearing_data) < max_visible_agents:
            hearing_data.append([0, 0])
        hearing_data = np.array(hearing_data[:max_visible_agents], dtype=np.float32)

        # 能量变化
        delta_energy = np.array([agent.energy - agent.last_energy], dtype=np.float32)
        agent.last_energy = agent.energy

        # 返回结构化的观察结果
        return {
            "observation": {
                "vision": vision_data,
                "hearing": hearing_data,
                "energy_change": delta_energy
            }
        }
class Predator(Agent):
    def __init__(self, position, velocity, energy=50, species=0, reproduction_intention=False):
        # 将 agent_type=1 显式传递给父类构造函数
        super().__init__(position, velocity, energy, species, agent_type=1, reproduction_intention=reproduction_intention)
        self.reproduction_probability = 0.5
        self.type = 1  # 设置捕食者类型为 1

    def interact_with_prey(self, prey_list):
        for i, prey in enumerate(prey_list):
            if np.linalg.norm(self.position - prey.position) < 10:
                self.energy += prey.energy
                return i
        return None

    def act_based_on_observation(self, observation):
        # 根据 species 获取对应的算法函数
        algorithm_function = self.environment.species_to_function.get(self.species, random_algorithm)
        
        # 调用算法函数生成动作
        action = algorithm_function(observation, normal_speed)
        return action
class Prey(Agent):
    def __init__(self, position, velocity, energy=50, species=0, reproduction_intention=False):
        # 将 agent_type=0 显式传递给父类构造函数
        super().__init__(position, velocity, energy, species, agent_type=0, reproduction_intention=reproduction_intention)
        self.type = 0  # 设置猎物类型为 0

    def interact_with_berries(self, berry_list):
        for i, berry in enumerate(berry_list):
            if np.linalg.norm(self.position - berry.position) < 10:
                self.energy += berry.energy
                return i
        return None

    def act_based_on_observation(self, observation):
        """
        根据观察信息生成动作。
        返回一个动作字典，而不直接修改智能体的状态。
        """
        # 默认不生育
        make_child = 0  

        # 随机决定是否生育
        reproduction_probability = 0.3  # 生育的概率
        if random.random() < reproduction_probability:
            make_child = 1

        # 随机生成速度
        angle = random.uniform(0, 2 * np.pi)
        speed = normal_speed
        velocity_x = np.cos(angle) * speed
        velocity_y = np.sin(angle) * speed

        # 返回动作字典
        return {
            "make_child": make_child,
            "velocity": [velocity_x, velocity_y]
        }

# Berry class
class Berry:
    def __init__(self, position, max_energy=100, decay_rate=2, reproduction_cost=50, growth_rate=1):
        self.position = position  # 食物位置
        self.energy = 1  # 初始能量
        self.max_energy = max_energy  # 最大能量
        self.decay_rate = decay_rate  # 衰退速率
        self.reproduction_cost = reproduction_cost  # 繁殖消耗的能量
        self.growth_rate = growth_rate  # 能量增长速率
        self.status = "growing"  # 初始状态是生长

    def update_energy(self):
        """
        根据当前状态更新食物的能量值，并处理状态转换。
        """
        if self.status == "growing":
            # 增加能量，直到达到最大能量
            self.energy = min(self.energy + self.growth_rate, self.max_energy)
            if self.energy == self.max_energy:
                self.status = "mature"  # 达到最大能量后变成熟
        elif self.status == "mature":
            # 能量已满，检查是否繁殖
            if self.energy >= self.reproduction_cost:
                self.status = "reproducing"  # 开始繁殖
        elif self.status == "reproducing":
            # 在繁殖状态，消耗能量并生成新的浆果
            self.energy -= self.reproduction_cost
            if self.energy <= 0:
                self.status = "decaying"  # 繁殖完成后进入衰退期
        elif self.status == "decaying":
            # 衰退状态，能量逐渐减少
            self.energy = max(self.energy - self.decay_rate, 0)
            if self.energy == 0:
                self.status = "dead"  # 生命力耗尽，死亡

    def get_color(self):
        """
        根据当前能量值返回颜色（渐变效果）。
        """
        energy_ratio = self.energy / self.max_energy
        if energy_ratio < 0.3:
            return (1, 0, 0)  # 红色：低能量，临近死亡
        elif energy_ratio < 0.6:
            return (1, 1, 0)  # 黄色：衰退期
        elif energy_ratio < 1.0:
            return (0, 1, 0)  # 绿色：成熟期
        else:
            return (0, 0, 1)  # 蓝色：生长期

from pettingzoo.utils.env import ParallelEnv
from gymnasium.spaces import Box, Discrete
import numpy as np
# from parallel_test import parallel_api_test

# 环境类
class Environment(ParallelEnv):
    def __init__(self, num_predators, num_prey, num_berries,species_to_function, max_visible_agents=5, num_species=2, max_steps=1000,render_mode=None):
        # 参数初始化
        self.render_mode = render_mode
        self.metadata = {
            "render_modes": ["human"],
            "name": "PredatorPreyParallelEnv"
        }
        self.species_to_function = species_to_function
        self.num_predators = num_predators
        self.num_prey = num_prey
        self.num_berries = num_berries
        self.max_visible_agents = max_visible_agents
        self.num_species = num_species
        self.agent_counter = 0  # 全局计数器
        self.max_steps = max_steps  # 最大步数
        self.current_step = 0  # 当前步数
        self.agent_counter = 0  # 全局计数器
        self.vision_radius = 200  # 感知半径
        self.vision_angle = np.pi  # 可视角度
        self.hearing_radius = 300  # 听觉半径
        self.max_visible_agents = 5  # 最大可见目标数量
        # 捕食者和猎物字典
        self.predators = {}
        self.preys = {}
        self.cached_observations= None
        # 强化学习智能体
        self.agents = []
        self.possible_agents = []
        self._agent_ids = set(self.possible_agents)
        # 初始化动作和观察空间


        # 初始化环境
        self.initialize_environment()
        # 智能体的动作空间和观察空间
        self.observation_spaces = {
            agent: self._create_observation_space() for agent in self.possible_agents
        }
        self.action_spaces = {
            agent: self._create_action_space() for agent in self.possible_agents
        }


        self.canvas = None
        self.view = None
        self.scatter_markers = None
        # self.scatter_prey = None
        # self.scatter_berries = None
        self.gui_enabled = False  # 控制GUI显示的标志
        self.predator_positions = np.zeros((0, 2))
        self.prey_positions = np.zeros((0, 2))
        self.berry_positions = np.zeros((0, 2))
        self.predator_colors = []
        self.prey_colors = []
        self.berry_colors = []
        # self._initialize_gui()
        self.fig = None
        # 绘图设置
        self.setup_plot()

    def _initialize_data(self):
        """初始化捕食者、猎物和浆果的数据"""
        self.predator_positions = np.array([p.position for p in self.predators.values()])
        self.prey_positions = np.array([p.position for p in self.preys.values()])
        self.berry_positions = np.array([b.position for b in self.berries])
        self.predator_colors = ['red'] * len(self.predator_positions)
        self.prey_colors = ['green'] * len(self.prey_positions)
        self.berry_colors = ['blue'] * len(self.berry_positions)
    def _initialize_gui(self):
        """
        初始化 GUI 对象（VisPy），仅在 render 时调用。
        """
        if self.canvas is None:  # 防止多次初始化
            self.canvas = scene.SceneCanvas(keys='interactive', size=(3840, 2160), show=True)
            self.canvas.bgcolor = 'white'
            self.view = self.canvas.central_widget.add_view()

        # 初始化 DynamicMarkers
        if self.scatter_markers is None:  # 确保只初始化一次
            self.scatter_markers = scene.visuals.Markers()
            self.view.add(self.scatter_markers)
        # 禁用GUI显示，待`render`控制
        # self.gui_enabled = False
    def initialize_environment(self):
        """
        初始化捕食者、猎物和浆果。
        """
        self.agent_counter = 0  
        self.possible_agents = []
        self.agents = []
        # 初始化捕食者
        for i in range(self.num_predators):
            species = i % self.num_species
            name = f"1_{species}_{self.agent_counter}"
            self.agent_counter += 1
            position = np.random.rand(2) * [3840, 2160]
            velocity = np.random.randn(2) * 2
            self.predators[name] = Predator(position, velocity, energy=50, species=species)
            self.possible_agents.append(name)

        # 初始化猎物
        for i in range(self.num_prey):
            species = i % self.num_species
            name = f"0_{species}_{self.agent_counter}"
            self.agent_counter += 1
            position = np.random.rand(2) * [3840, 2160]
            velocity = np.random.randn(2) * 2
            self.preys[name] = Prey(position, velocity, energy=50, species=species)
            self.possible_agents.append(name)

        # 初始化浆果
        self.berries = [Berry(position=np.random.rand(2) * [3840, 2160]) for _ in range(self.num_berries)]
        self.agents = self.possible_agents.copy()
        self._agent_ids =self.possible_agents.copy()
        # 初始化统计变量
        self.reset_statistics()
    def reset_statistics(self):
        """
        重置环境中的能量和数量统计变量。
        """
        self.total_energy = []        # 系统总能量
        self.predator_energy = []     # 捕食者总能量
        self.prey_energy = []         # 猎物总能量
        self.berry_energy = []        # 浆果总能量
        self.predator_count = []      # 捕食者数量
        self.prey_count = []          # 猎物数量
        self.berry_count = []         # 浆果数量
    def _create_observation_space(self):
        """
        定义观察空间：
        包括视觉观测、听觉观测和能量变化。
        """
        return Dict({
            "observation": Dict({
                "vision": Box(
                    low=np.array([[-1.0, -1.0, 0, -1, -2]] * self.max_visible_agents),
                    high=np.array([[1.0, 1.0, 1, self.num_species - 1, 1]] * self.max_visible_agents),
                    shape=(self.max_visible_agents, 5),
                    dtype=np.float32
                ),
                "hearing": Box(
                    low=np.array([[0, 0.0]] * self.max_visible_agents),
                    high=np.array([[1, 1.0]] * self.max_visible_agents),
                    shape=(self.max_visible_agents, 2),
                    dtype=np.float32
                ),
                "energy_change": Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(1,),
                    dtype=np.float32
                ),
            })})

    def _create_action_space(self):
        """
        定义动作空间：
        包括生育意图（离散值）和速度（连续值）。
        """
        return Dict({
            "make_child": Discrete(2),  # 生育意图：0 或 1
            "velocity": Box(
                low=-1.0,
                high=1.0,
                shape=(2,),
                dtype=np.float32
            ),
        })
    def _batch_observe(self):
        """
        批量生成环境中所有智能体的观测信息。

        返回:
            - observations: 字典，键是智能体名字，值是对应的观测信息。
        """
        agents = {**self.predators, **self.preys}  # 捕食者和猎物的字典合并
        agent_names = list(agents.keys())
        agent_positions = np.array([agent.position for agent in agents.values()])
        berry_positions = np.array([berry.position for berry in self.berries])

        # 确保 agent_positions 和 berry_positions 至少是空的二维数组
        if len(agent_positions) == 0:
            agent_positions = np.zeros((0, 2), dtype=np.float32)
        if len(berry_positions) == 0:
            berry_positions = np.zeros((0, 2), dtype=np.float32)

        # 构建 KDTree
        all_positions = np.vstack([agent_positions, berry_positions])  # 保证至少是二维空数组
        if len(all_positions) == 0:
            # 如果没有任何智能体或浆果，直接返回空观测
            return {}

        tree = KDTree(all_positions)

        # 返回结果的字典
        observations = {}

        # 遍历每个智能体生成其观测
        for idx, agent_name in enumerate(agent_names):
            agent = agents[agent_name]

            # 查询视觉范围内的目标
            neighbors_indices = tree.query_ball_point(agent.position, r=self.vision_radius)

            # 初始化视觉数据
            vision_data = []
            for neighbor_idx in neighbors_indices:
                # 跳过自身
                if neighbor_idx == idx:
                    continue

                relative_position = all_positions[neighbor_idx] - agent.position
                dx = relative_position[0] / self.vision_radius  # 归一化 x
                dy = relative_position[1] / self.vision_radius  # 归一化 y

                if neighbor_idx < len(agent_positions):  # 目标是智能体
                    target_agent = agents[agent_names[neighbor_idx]]
                    reproduction = int(target_agent.reproduction_intention)
                    species = target_agent.species
                    agent_type = 1 if isinstance(target_agent, Predator) else -1
                else:  # 目标是浆果
                    reproduction = 0
                    species = -1
                    agent_type = -2

                vision_data.append([dx, dy, reproduction, species, agent_type])

            # 填充默认值
            while len(vision_data) < self.max_visible_agents:
                vision_data.append([0, 0, 0, 0, 0])
            vision_data = np.array(vision_data[:self.max_visible_agents], dtype=np.float32)

            # 查询听觉范围内的目标
            hearing_data = []
            hearing_neighbors = tree.query_ball_point(agent.position, r=self.hearing_radius)
            for neighbor_idx in hearing_neighbors:
                if neighbor_idx == idx:
                    continue

                relative_position = all_positions[neighbor_idx] - agent.position
                distance = np.linalg.norm(relative_position)
                direction = 0 if relative_position[0] < 0 else 1
                intensity = 1 - (distance / self.hearing_radius)
                hearing_data.append([direction, intensity])

            # 填充听觉默认值
            while len(hearing_data) < self.max_visible_agents:
                hearing_data.append([0, 0])
            hearing_data = np.array(hearing_data[:self.max_visible_agents], dtype=np.float32)

            # 能量变化
            delta_energy = np.array([agent.energy - agent.last_energy], dtype=np.float32)
            agent.last_energy = agent.energy

            # 存储观测信息
            observations[agent_name] = {
                "observation": {
                    "vision": vision_data,
                    "hearing": hearing_data,
                    "energy_change": delta_energy
                }
            }
        self.cached_observations = observations
        return observations
    def reset(self, seed=None, options=None):
        """
        重置环境到初始状态。
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        # 重置环境
        self.initialize_environment()
        self._initialize_data()
        # 初始化观察和信息
        self._batch_observe()

        # 筛选出当前激活的智能体
        observations = {agent: self.cached_observations[agent] for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        assert set(self.agents).issubset(self.possible_agents), f"Agents are not in possible_agents{len(self.agents),{len(self.possible_agents)}}"
        if len(self.agents) != len(self.possible_agents):
            print("<><><><><>",len(self.agents),len(self.possible_agents))
        return observations, infos

    def _generate_initial_observation(self, agent):
        """
        根据智能体类型生成初始观察值，并记录 observe 函数的执行时间。
        """
        # 判断智能体类型
        if agent in self.predators:
            agent_object = self.predators[agent]

            # 将猎物和浆果信息转换为 NumPy 数组以便后续优化
            prey_list = list(self.preys.values())

            # 记录 observe 执行时间
            # start_time = time.time()
            observation = agent_object.observe(prey_list, self.berries)
            # end_time = time.time()
            # print(f"observe execution time for predator {agent}: {end_time - start_time:.6f} seconds")

            return observation

        elif agent in self.preys:
            agent_object = self.preys[agent]

            # 将捕食者和浆果信息转换为 NumPy 数组以便后续优化
            predator_list = list(self.predators.values())

            # 记录 observe 执行时间
            # start_time = time.time()
            observation = agent_object.observe(predator_list, self.berries)
            # end_time = time.time()
            # print(f"observe execution time for prey {agent}: {end_time - start_time:.6f} seconds")

            return observation

        else:
            # 未知智能体
            raise ValueError(f"Unknown agent name: {agent}")

    def render(self, mode="human"):
        """
        渲染环境。延迟初始化 GUI，仅在需要时激活。
        """
        if not self.gui_enabled:
            # 初始化 GUI
            self.canvas = scene.SceneCanvas(keys='interactive', size=(3840, 2160), show=True)
            self.canvas.bgcolor = 'white'
            self.view = self.canvas.central_widget.add_view()

            # 创建单一的 DynamicMarkers 实例
            self.scatter_markers = scene.visuals.Markers()
            self.view.add(self.scatter_markers)
            self._initialize_gui()  # 调用 GUI 初始化方法
            self.gui_enabled = True

        # 更新数据并渲染
        self._update_data()  # 确保数据已更新
        self._batch_update_visualization()  # 更新可视化
        self.update_plots()  # 更新绘图
    def _update_data(self):
        """仅更新捕食者、猎物和浆果的变化部分"""
        # 捕食者位置
        self.predator_positions = np.array([p.position for p in self.predators.values()])
        self.predator_colors = ['red'] * len(self.predator_positions)

        # 猎物位置
        self.prey_positions = np.array([p.position for p in self.preys.values()])
        self.prey_colors = ['green'] * len(self.prey_positions)

        # 浆果位置
        self.berry_positions = np.array([b.position for b in self.berries])
        self.berry_colors = ['blue'] * len(self.berry_positions)
    def close(self):
        """
        关闭环境并释放资源。
        """
        if hasattr(self, 'fig') and self.fig is not None:
            plt.close(self.fig)
            self.fig = None
        print("Closing environment...")

    def observation_space(self, agent):
        """
        返回指定智能体的观察空间。
        """
        return self.observation_spaces[agent]

    def action_space(self, agent):
        """
        返回指定智能体的动作空间。
        """
        return self.action_spaces[agent]
    def setup_plot(self):

        if hasattr(self, 'fig') and self.fig is not None:  # 如果窗口已存在，则不重复创建
            return
        # Initialize the plotting window with three subplots
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(3, 1, figsize=(10, 12))
        
        # Set titles and labels
        self.ax1.set_title("System Total Energy Over Time")
        self.ax1.set_xlabel("Time Steps")
        self.ax1.set_ylabel("Total Energy")
        
        self.ax2.set_title("Energy by Group Over Time")
        self.ax2.set_xlabel("Time Steps")
        self.ax2.set_ylabel("Energy")
        
        self.ax3.set_title("Agent Population Over Time")
        self.ax3.set_xlabel("Time Steps")
        self.ax3.set_ylabel("Population Count")
        
        plt.ion()  # Enable interactive mode
        plt.show()

    def update_plots(self):
        if not hasattr(self, 'fig') or self.fig is None:
            self.setup_plot()  # 如果未初始化绘图窗口，则初始化

        # 清空之前的绘图
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()

        # 绘制系统总能量
        total_energy = self.total_energy if len(self.total_energy) > 0 else [0]
        self.ax1.plot(total_energy, label="Total Energy", color='purple')
        self.ax1.legend()

        # 绘制每个群体的能量
        predator_energy = self.predator_energy if len(self.predator_energy) > 0 else [0]
        prey_energy = self.prey_energy if len(self.prey_energy) > 0 else [0]
        berry_energy = self.berry_energy if len(self.berry_energy) > 0 else [0]
        self.ax2.plot(predator_energy, label="Predator Energy", color='red')
        self.ax2.plot(prey_energy, label="Prey Energy", color='green')
        self.ax2.plot(berry_energy, label="Berry Energy", color='blue')
        self.ax2.legend()

        # 绘制群体数量
        predator_count = self.predator_count if len(self.predator_count) > 0 else [0]
        prey_count = self.prey_count if len(self.prey_count) > 0 else [0]
        berry_count = self.berry_count if len(self.berry_count) > 0 else [0]
        self.ax3.plot(predator_count, label="Predator Count", color='red')
        self.ax3.plot(prey_count, label="Prey Count", color='green')
        self.ax3.plot(berry_count, label="Berry Count", color='blue')
        self.ax3.legend()

        # 更新绘图
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
     
    def step(self, actions):
        """Execute a step in the environment."""
        # 更新环境状态
        self.update(actions)

        # 获取强化学习框架所需的输出
        observations, rewards, terminated, truncations, infos = self._process_agents()

        return observations, rewards, terminated, truncations, infos


    def _process_agents(self):
        observations, rewards, terminated, infos = {}, {}, {}, {}
        self.current_step += 1

        # 设置截断标志
        truncations = {agent: (self.current_step >= self.max_steps) for agent in self.agents}

        # zero_data = {
        #     'observation': {
        #         'vision': np.zeros((self.max_visible_agents, 5), dtype=np.float32),
        #         'hearing': np.zeros((self.max_visible_agents, 2), dtype=np.float32),
        #         'energy_change': np.array([0.0], dtype=np.float32)  # 确保 shape=(1,)
        #     }
        # }
        live_agents = [agent for agent in self.agents if agent in self.preys or agent in self.predators]
        for agent in self.agents:
            if agent in live_agents:
                terminated[agent] = False
            elif agent not in live_agents:
                terminated[agent] = True 
        # terminations = {
        #     agent: int(life) < 0
        #     for agent, life in zip(self.possible_agents, lives)
        #     if agent in self.agents
        # }
        for agent in self.agents:
            
            observations[agent] = {
            'observation': {
                'vision': np.zeros((self.max_visible_agents, 5), dtype=np.float32),
                'hearing': np.zeros((self.max_visible_agents, 2), dtype=np.float32),
                'energy_change': np.array([0.0], dtype=np.float32)  # 确保 shape=(1,)
            }}
            observations[agent] = self.cached_observations[agent]
            rewards[agent] = 0.0
            rewards[agent] = self.cached_observations[agent]['observation']["energy_change"].item()
            # if agent[0] =="1":
                
            #     observations[agent] = self.predators[agent].observe(self.preys.values(), self.berries)
            #     rewards[agent] = self.predators[agent].observe(self.preys.values(), self.berries)['observation']["energy_change"].item()
            # elif agent[1] =='0':
            #     observations[agent] = self.predators[agent].observe(self.predators.values(), self.berries)
            #     rewards[agent] = self.predators[agent].observe(self.preys.values(), self.berries)['observation']["energy_change"].item()                
        infos = {agent: {} for agent in self.possible_agents if agent in self.agents}
        self.agents = [agent for agent in self.agents if not terminated[agent]]
        return observations, rewards, terminated, truncations, infos

    def initialize_agents(self, agent_class, num_agents, agent_type):
        agents = {}
        current_index = 0  # 用于分配智能体的全局计数
        for i in range(num_agents):  # 总共创建 num_agents 个智能体
            species = current_index % self.num_species  # 按顺序分配物种
            current_index += 1
            
            position = np.random.rand(2) * [width, height]
            velocity = np.random.randn(2) * 2
            energy = random.randint(40, 60)

            # 初始化智能体，自动生成名字
            agent = agent_class(position, velocity, energy, species, agent_type)
            agents[agent.name] = agent  # 将名字作为键，智能体作为值存储
        return agents

    def update(self, actions=None):
        """
        更新环境状态：
        - 根据动作字典更新捕食者和猎物的状态
        - 处理捕食、觅食和繁殖逻辑
        - 动态移除死亡智能体并更新统计信息
        """
        # 初始化动作字典（如果未提供）
        actions = actions or {}
        self._batch_observe()
        # 遍历所有存活的智能体，生成完整的动作字典
        # 为未提供动作的智能体生成默认动作
        for agent_name in self.agents:
            if agent_name not in actions:
                # 使用缓存的观测信息生成动作
                observation = self.cached_observations[agent_name]
                if agent_name in self.predators:
                    actions[agent_name] = self.predators[agent_name].act_based_on_observation(observation)
                elif agent_name in self.preys:
                    actions[agent_name] = self.preys[agent_name].act_based_on_observation(observation)

        # 更新捕食者状态
        self._update_predators(actions)

        # 更新猎物状态
        self._update_prey(actions)    
        # 处理繁殖逻辑
        self.reproduce() 
        # 更新浆果状态
        self._update_berries()

        # 更新统计数据
        self._update_statistics()

        # 更新可视化数据
        # self._update_visualization()

        # 移除已死亡的智能体
        # self.agents = list(self.predators.keys()) + list(self.preys.keys())   
        # self.agents = [agent for agent in self.agents if agent in self.predators and self.predators[agent].energy > 0] + \
        #             [agent for agent in self.agents if agent in self.preys and self.preys[agent].energy > 0]
    def _update_predators(self, actions):
        """
        更新捕食者的状态和行为。
        """
        for name, predator in self.predators.items():
            action = actions.get(name, {"make_child": 0, "velocity": [0.0, 0.0]})

            # 更新捕食者状态
            predator.velocity = np.array(action["velocity"]) * normal_speed
            predator.reproduction_intention = bool(action["make_child"])

            # 捕食行为：尝试捕获猎物
            prey_index = predator.interact_with_prey(list(self.preys.values()))
            if prey_index is not None:
                prey_name = list(self.preys.keys())[prey_index]
                predator.energy += self.preys[prey_name].energy  # 捕食增加能量
                del self.preys[prey_name]  # 移除被捕食的猎物

            # 移动捕食者
            predator.move()
    def _update_prey(self, actions):
        """
        更新猎物的状态和行为。
        """
        for name, prey in self.preys.items():
            action = actions.get(name, {"make_child": 0, "velocity": [0.0, 0.0]})

            # 更新猎物状态
            prey.velocity = np.array(action["velocity"]) * normal_speed
            prey.reproduction_intention = bool(action["make_child"])

            # 觅食行为：尝试吃浆果
            berry_index = prey.interact_with_berries(self.berries)
            if berry_index is not None:
                prey.energy += berry_energy  # 吃浆果增加能量
                del self.berries[berry_index]  # 移除被食用的浆果

            # 移动猎物
            prey.move()    
    def _update_berries(self):
        """
        更新浆果的状态，包括生命值的变化、繁殖和衰退。
        """
        new_berries = []
        for berry in self.berries:
            berry.update_energy()  # 更新浆果的能量和状态

            # 如果浆果已经死亡，则移除
            if berry.status == "dead":
                continue

            # 如果浆果处于成熟状态并且能量充足，尝试繁殖
            if berry.status == "mature" and berry.energy >= berry.reproduction_cost:
                new_position = berry.position + np.random.uniform(-berry_reproduction_radius, berry_reproduction_radius, 2)
                new_berries.append(Berry(new_position))

        # 将新生成的浆果添加到现有的浆果列表中
        self.berries.extend(new_berries)
    def _update_statistics(self):
        """
        统计当前能量和智能体数量。
        """
        predator_total_energy = sum([p.energy for p in self.predators.values()])
        prey_total_energy = sum([p.energy for p in self.preys.values()])
        berry_total_energy = sum([b.energy for b in self.berries])

        self.total_energy.append(predator_total_energy + prey_total_energy + berry_total_energy)
        self.predator_energy.append(predator_total_energy)
        self.prey_energy.append(prey_total_energy)
        self.berry_energy.append(berry_total_energy)
        self.predator_count.append(len(self.predators))
        self.prey_count.append(len(self.preys))
        self.berry_count.append(len(self.berries))
    # def _async_update_visualization(self):
    #     """使用线程异步更新可视化。"""
    #     def update():
    #         predator_positions = np.array([p.position for p in self.predators.values()])
    #         prey_positions = np.array([p.position for p in self.preys.values()])
    #         berry_positions = np.array([b.position for b in self.berries])

    #         # 批量绘制数据
    #         self.scatter_predators.set_data(
    #             np.vstack([predator_positions, prey_positions, berry_positions]),
    #             edge_color=['red'] * len(predator_positions) +
    #                     ['green'] * len(prey_positions) +
    #                     ['blue'] * len(berry_positions),
    #             face_color=['red'] * len(predator_positions) +
    #                     ['green'] * len(prey_positions) +
    #                     ['blue'] * len(berry_positions),
    #             size=10
    #         )
    #         self.update_plots()

    #     # 启动一个新线程处理可视化更新
    #     Thread(target=update).start()
    def _batch_update_visualization(self):
        """批量更新可视化数据"""
        # 如果某些群体为空，确保位置数组至少是空的二维数组
        predator_positions = self.predator_positions if len(self.predator_positions) > 0 else np.zeros((0, 2), dtype=np.float32)
        prey_positions = self.prey_positions if len(self.prey_positions) > 0 else np.zeros((0, 2), dtype=np.float32)
        berry_positions = self.berry_positions if len(self.berry_positions) > 0 else np.zeros((0, 2), dtype=np.float32)

        # 汇总所有物体的位置
        all_positions = np.vstack([predator_positions, prey_positions, berry_positions])

        # 汇总所有物体的颜色
        predator_colors = self.predator_colors if len(self.predator_colors) > 0 else []
        prey_colors = self.prey_colors if len(self.prey_colors) > 0 else []
        berry_colors = self.berry_colors if len(self.berry_colors) > 0 else []
        all_colors = predator_colors + prey_colors + berry_colors

        # 如果 all_positions 或 all_colors 为空，填充默认值
        if len(all_positions) == 0:
            all_positions = np.zeros((1, 2), dtype=np.float32)  # 至少保留一个点
            all_colors = [(0.5, 0.5, 0.5, 1.0)]  # 灰色表示无物体

        # 更新散点图中的数据
        self.scatter_markers.set_data(
            all_positions,          # 所有物体的位置
            edge_color=all_colors,  # 物体边缘颜色
            face_color=all_colors,  # 物体填充颜色
            size=10                 # 物体大小
        )
    # def _update_visualization(self):
    #     """使用批量绘制优化可视化。"""
    #     # 合并所有点的位置和颜色
    #     all_positions = []
    #     all_colors = []

    #     # 捕食者
    #     predator_positions = np.array([p.position for p in self.predators.values()])
    #     all_positions.append(predator_positions)
    #     all_colors.extend(['red'] * len(predator_positions))

    #     # 猎物
    #     prey_positions = np.array([p.position for p in self.preys.values()])
    #     all_positions.append(prey_positions)
    #     all_colors.extend(['green'] * len(prey_positions))

    #     # 浆果
    #     berry_positions = np.array([b.position for b in self.berries])
    #     all_positions.append(berry_positions)
    #     all_colors.extend(['blue'] * len(berry_positions))

    #     # 批量设置数据
    #     all_positions = np.vstack(all_positions)
    #     self.scatter_predators.set_data(
    #         all_positions,
    #         edge_color=all_colors,
    #         face_color=all_colors,
    #         size=10
    #     )

    #     # 更新图表
    #     self.update_plots()


    def plot_data(self):
            # 设置绘图窗口
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

            # 系统总能量图
            ax1.set_title("System Total Energy Over Time")
            ax1.set_xlabel("Time Steps")
            ax1.set_ylabel("Total Energy")
            
            # 各群体能量图
            ax2.set_title("Energy by Group Over Time")
            ax2.set_xlabel("Time Steps")
            ax2.set_ylabel("Energy")
            
            # 更新函数
            def animate(i):
                if i < len(self.total_energy):
                    ax1.clear()
                    ax2.clear()

                    # 绘制总能量
                    ax1.plot(self.total_energy[:i], label="Total Energy", color='purple')
                    ax1.legend()

                    # 绘制群体能量
                    ax2.plot(self.predator_energy[:i], label="Predator Energy", color='red')
                    ax2.plot(self.prey_energy[:i], label="Prey Energy", color='green')
                    ax2.plot(self.berry_energy[:i], label="Berry Energy", color='blue')
                    ax2.legend()

            ani = FuncAnimation(fig, animate, frames=len(self.total_energy), interval=100, repeat=False)
            plt.show()
    def reproduce(self):
        """
        捕食者和猎物的繁殖逻辑。
        """
        self.reproduce_species(agents=self.predators, species_key="predator", agent_class=Predator)
        self.reproduce_species(agents=self.preys, species_key="prey", agent_class=Prey)
    def reproduce_species(self,agents, species_key, agent_class):
        """
        通用繁殖逻辑，用于捕食者和猎物。
        - agents: 当前智能体的字典
        - species_key: 智能体的名字前缀（例如 'predator' 或 'prey'）
        - agent_class: 智能体类（Predator 或 Prey）
        """
        if not agents:
            return

        new_agents = {}
        energy_threshold = 2000000000000000  # 繁殖能量阈值

        # 使用列表存储智能体位置和名字，用于加速邻居搜索
        positions = np.array([agent.position for agent in agents.values()])
        if positions.size == 0:
            return
        names = list(agents.keys())

        # 使用 KDTree 优化邻居搜索
        tree = KDTree(positions)

        for i, (name, agent) in enumerate(agents.items()):
            # 类型和属性检查
            assert isinstance(agent.position, np.ndarray), f"Agent {name} position must be a numpy array."
            if not hasattr(agent, "reproduction_intention"):
                agent.reproduction_intention = False

            # 跳过能量不足或无繁殖意图的智能体
            if agent.energy <= energy_threshold or not agent.reproduction_intention:
                continue

            # 搜索附近的同类智能体
            neighbor_indices = tree.query_ball_point(agent.position, r=10)
            if not neighbor_indices:
                continue

            for j in neighbor_indices:
                other_name = names[j]
                other_agent = agents[other_name]

                # 确保不同对象且满足繁殖条件
                if name != other_name and other_agent.species == agent.species:
                    if other_agent.reproduction_intention and other_agent.energy > energy_threshold:
                        # 繁殖：生成新智能体
                        new_position = (agent.position + other_agent.position) / 2
                        new_position += np.random.uniform(-position_perturbation, position_perturbation, 2)
                        
                        energy_transfer = agent.energy / 3
                        agent.energy -= energy_transfer
                        other_agent.energy -= energy_transfer
                        new_energy = energy_transfer * 2

                        # 确保生成的新名字唯一
                        new_name = f"{species_key}_{self.agent_counter}"
                        while new_name in agents:
                            self.agent_counter += 1
                            new_name = f"{species_key}_{self.agent_counter}"

                        # 创建新智能体
                        new_agent = agent_class(
                            position=new_position,
                            velocity=np.random.randn(2) * velocity_perturbation,
                            energy=new_energy,
                            species=agent.species
                        )
                        new_agents[new_name] = new_agent

                        # 调整父母的位置和速度
                        agent.position += np.random.uniform(-position_perturbation, position_perturbation, 2)
                        other_agent.position += np.random.uniform(-position_perturbation, position_perturbation, 2)
                        agent.velocity += np.random.randn(2) * velocity_perturbation
                        other_agent.velocity += np.random.randn(2) * velocity_perturbation

        # 将新生成的智能体加入环境
        agents.update(new_agents)


def verify_reset_observation(environment):
    """
    验证 reset 函数生成的观察是否与智能体的观察空间一致。
    """
    # 调用 reset 方法
    observations, infos = environment.reset()
    
    # 遍历所有智能体
    for agent in environment.agents:
        # 获取智能体的观察空间
        obs_space = environment.observation_space(agent)
        
        # 获取智能体的初始观察
        observation = observations[agent]
        
        # 验证观察是否符合观察空间
        if not obs_space.contains(observation):
            print(f"Observation for agent {agent} does not match the observation space!")
            print(f"Observation: {observation}")
            print(f"Observation space: {obs_space.sample()}")
            return False
    
    print("All observations match the observation space!")
    return True

# from parallel_test import parallel_api_test
# from api_test import api_test
from pettingzoo.utils import parallel_to_aec
# 实例化环境

def random_algorithm(observation_info, max_speed):
    make_child = np.random.choice([0, 1])
    angle = np.random.uniform(0, 2 * np.pi)
    length = np.random.uniform(0, max_speed)
    x = length * np.cos(angle)
    y = length * np.sin(angle)
    return {"make_child": make_child, "velocity": np.array([x, y], dtype=np.float32)}

def ppo_algorithm(observation_info, max_speed):
    # Placeholder PPO logic (random for now)
    return random_algorithm(observation_info, max_speed)

def sac_algorithm(observation_info, max_speed):
    # Placeholder SAC logic (random for now)
    return random_algorithm(observation_info, max_speed)

species_to_function = {
    0: random_algorithm,  # species 0 使用 random_algorithm
    1: ppo_algorithm,     # species 1 使用 ppo_algorithm
    2: sac_algorithm      # species 2 使用 sac_algorithm
}
from pettingzoo.test import api_test
# environment = Environment(num_predators=1, num_prey=30, num_berries=500, species_to_function=species_to_function)
# api_test(environment)
# print(123)



from gymnasium import spaces
import numpy as np
from pettingzoo.utils.env import ParallelEnv

def flatten_pettingzoo_env(env_class):
    """
    接受一个 PettingZoo 环境类，返回一个新的类，其中观测信息被展平。
    """
    class FlattenedEnv(env_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # 对所有 agent 的观察空间进行展平
            self.observation_spaces = {
                agent_name: self._flatten_space(space)
                for agent_name, space in self.observation_spaces.items()
            }

        def reset(self, seed=None, options=None):
            """
            重置环境，并返回展平的观测信息。
            """
            observations, infos = super().reset(seed=seed, options=options)
            flattened_observations = {
                agent_name: self._flatten_observation(obs)
                for agent_name, obs in observations.items()
            }
            return flattened_observations, infos

        def step(self, actions):
            """
            执行一步，并返回展平的观测信息。
            """
            observations, rewards, terminated, truncations, infos = super().step(actions)
            flattened_observations = {
                agent_name: self._flatten_observation(obs)
                for agent_name, obs in observations.items()
            }
            return flattened_observations, rewards, terminated, truncations, infos

        @staticmethod
        def _flatten_observation(observation):
            """
            将输入的观测数据展平为一维数组。
            """
            if isinstance(observation, dict):
                return np.concatenate([FlattenedEnv._flatten_observation(v) for v in observation.values()])
            elif isinstance(observation, (tuple, list)):
                return np.concatenate([FlattenedEnv._flatten_observation(v) for v in observation])
            elif isinstance(observation, np.ndarray):
                return observation.flatten()
            else:
                return np.array([observation], dtype=np.float32)

        @staticmethod
        def _flatten_space(space):
            """
            将 Gym 空间展平为一维 Box 空间。
            """
            if isinstance(space, spaces.Dict):
                flattened_spaces = [FlattenedEnv._flatten_space(s) for s in space.spaces.values()]
                total_dim = sum(flat.shape[0] for flat in flattened_spaces)
                return spaces.Box(low=-np.inf, high=np.inf, shape=(total_dim,), dtype=np.float32)
            elif isinstance(space, spaces.Tuple):
                flattened_spaces = [FlattenedEnv._flatten_space(s) for s in space.spaces]
                total_dim = sum(flat.shape[0] for flat in flattened_spaces)
                return spaces.Box(low=-np.inf, high=np.inf, shape=(total_dim,), dtype=np.float32)
            elif isinstance(space, spaces.Box):
                return spaces.Box(
                    low=np.concatenate([space.low.flatten()]),
                    high=np.concatenate([space.high.flatten()]),
                    dtype=np.float32
                )
            elif isinstance(space, spaces.Discrete):
                return spaces.Box(low=0, high=space.n - 1, shape=(1,), dtype=np.float32)
            elif isinstance(space, spaces.MultiBinary):
                return spaces.Box(low=0, high=1, shape=(space.n,), dtype=np.float32)
            elif isinstance(space, spaces.MultiDiscrete):
                total_dim = sum(space.nvec)
                return spaces.Box(low=0, high=1, shape=(total_dim,), dtype=np.float32)
            else:
                raise NotImplementedError(f"Unsupported space type: {type(space)}")

    return FlattenedEnv
import supersuit as ss
FlattenedEnvironment = flatten_pettingzoo_env(Environment)
environment = Environment(num_predators=0, num_prey=300, num_berries=100, species_to_function=species_to_function)

environment1 = FlattenedEnvironment(num_predators=0, num_prey=1, num_berries=5000, species_to_function=species_to_function)
gym_env = ss.pettingzoo_env_to_vec_env_v1(environment1)
import pickle
try:
    pickle.dumps(gym_env)
except Exception as e:
    print(f"Pickle error: {e}")
gym_env = ss.concat_vec_envs_v1(gym_env, num_vec_envs=4, num_cpus=1, base_class='gymnasium')

import os

import ray
import supersuit as ss
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.tune.registry import register_env
from torch import nn

from pettingzoo.butterfly import pistonball_v6


class CNNModelV2(TorchModelV2, nn.Module):
    def __init__(self, obs_space, act_space, num_outputs, *args, **kwargs):
        TorchModelV2.__init__(self, obs_space, act_space, num_outputs, *args, **kwargs)
        nn.Module.__init__(self)
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, [8, 8], stride=(4, 4)),
            nn.ReLU(),
            nn.Conv2d(32, 64, [4, 4], stride=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(64, 64, [3, 3], stride=(1, 1)),
            nn.ReLU(),
            nn.Flatten(),
            (nn.Linear(3136, 512)),
            nn.ReLU(),
        )
        self.policy_fn = nn.Linear(512, num_outputs)
        self.value_fn = nn.Linear(512, 1)

    def forward(self, input_dict, state, seq_lens):
        model_out = self.model(input_dict["obs"].permute(0, 3, 1, 2))
        self._value_out = self.value_fn(model_out)
        return self.policy_fn(model_out), state

    def value_function(self):
        return self._value_out.flatten()


def env_creator(args):
    env = environment1
    # env = ss.color_reduction_v0(env, mode="B")
    env = ss.dtype_v0(env, "float32")
    # env = ss.resize_v1(env, x_size=84, y_size=84)
    # env = ss.normalize_obs_v0(env, env_min=0, env_max=1)
    # env = ss.frame_stack_v1(env, 3)
    return env


if __name__ == "__main__":
    ray.init()

    env_name = "FlattenedEnvironment"

    register_env(env_name, lambda config: ParallelPettingZooEnv(env_creator(config)))
    ModelCatalog.register_custom_model("CNNModelV2", CNNModelV2)

    config = (
        PPOConfig()
        .environment(env=env_name, clip_actions=True,disable_env_checking=True)
        .rollouts(num_rollout_workers=4, rollout_fragment_length=128)
        .training(
            train_batch_size=512,
            lr=2e-5,
            gamma=0.99,
            lambda_=0.9,
            use_gae=True,
            clip_param=0.4,
            grad_clip=None,
            entropy_coeff=0.1,
            vf_loss_coeff=0.25,
            sgd_minibatch_size=64,
            num_sgd_iter=10,
        )
        .debugging(log_level="ERROR")
        .framework(framework="torch")
        .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
    )

    tune.run(
        "PPO",
        name="PPO",
        stop={"timesteps_total": 5000000 if not os.environ.get("CI") else 50000},
        checkpoint_freq=10,
        local_dir="/home/qrbao/Downloads/code/code2/2predatorpreygame/" + env_name,
        config=config.to_dict(),
    )

from ray.tune.registry import _global_registry, ENV_CREATOR

# 列出所有注册的环境
registered_envs = list(_global_registry.get(ENV_CREATOR).keys())
print("Registered environments in Ray:", registered_envs)












# import os
# import ray
# from ray.tune.registry import register_env
# from ray.rllib.algorithms.ppo import PPO, PPOConfig
# from ray.rllib.env import ParallelPettingZooEnv

# # Make sure to use the same environment creation function
# def env_creator(args):
#     env = environment1
#     # env = ss.color_reduction_v0(env, mode="B")
#     env = ss.dtype_v0(env, "float32")
#     # env = ss.resize_v1(env, x_size=84, y_size=84)
#     # env = ss.normalize_obs_v0(env, env_min=0, env_max=1)
#     # env = ss.frame_stack_v1(env, 3)
#     return env
# def action_random(environment):
#     """
#     随机为每个存活的智能体生成动作。
#     """
#     actions = {}
#     for agent_name in environment.agents:
#         # 检查动作空间是否存在
#         if agent_name in environment.action_spaces:
#             actions[agent_name] = environment.action_space(agent_name).sample()
#         else:
#             # 跳过没有定义动作空间的智能体
#             # print(f"Skipping action generation for agent: {agent_name} (no action space found)")
#             pass
#     return actions
# if __name__ == "__main__":
#     ray.init()

#     # Register the environment
#     env_name = "FlattenedEnvironment"
#     register_env(env_name, lambda config: ParallelPettingZooEnv(env_creator(config)))

#     # Path to the saved checkpoint (replace with your actual checkpoint path)
#     # checkpoint_path = os.path.expanduser("/home/qrbao/ray_results/LISPredatorPreyFlattenedEnv/PPO/PPO_LISPredatorPreyFlattenedEnv_faf3c_00000_0_2024-10-24_18-18-50/checkpoint_000097")
#     # checkpoint_path = os.path.expanduser("/home/qrbao/ray_results/LISPredatorPreyFlattenedEnv/PPO/PPO_LISPredatorPreyFlattenedEnv_8fe80_00000_0_2024-10-25_15-01-22/checkpoint_000097")
#     # checkpoint_path = os.path.expanduser("/home/qrbao/Downloads/code/code2/2predatorpreygame/FlattenedEnvironment/PPO/PPO_FlattenedEnvironment_9ed84_00000_0_2024-11-26_14-38-09/checkpoint_000000")
#     checkpoint_path = os.path.expanduser("/home/qrbao/Downloads/code/code2/2predatorpreygame/FlattenedEnvironment/PPO/PPO_FlattenedEnvironment_26318_00000_0_2024-11-30_18-40-35/checkpoint_000008")
    



#     # Create a PPO configuration
#     config = (
#         PPOConfig()
#         .environment(env=env_name,disable_env_checking=True)
#         .framework("torch")
#     )

#     # Initialize the PPO algorithm with the configuration
#     ppo_agent = PPO(config=config)
#     # Restore from checkpoint
#     ppo_agent.restore(checkpoint_path)

#     # Run evaluation with the restored model
#     env = env_creator({})
#     """Run a random simulation with the environment."""
#     obs, info = env.reset()

#     data_storage = {
#         'iterations': [],
#         'predator_counts': [],
#         'prey_counts': [],
#         'total_counts': [],
#         'predator_healths': [],
#         'prey_healths': [],
#         'total_healths': []
#     }
#     done = False
#     total_reward = 0
#     plt.figure(figsize=(10, 8))
#     plt.ion()
#     game_continue = False
#     iteration = 0

#     while True:
#         env.render()
#         actions = {}
#         # if iteration % 100 == 1:
#             # update_and_plot(iteration, env, data_storage)
#             # print(len(env.simulator.predators),end="\t")
#             # for agent in env.simulator.predators + env.simulator.preys:
#             #     print(f"{agent.name} health is {agent.health}-||||||",end="\t")
#             # print(len(env.simulator.preys))
#         for agent_id, agent_obs in obs.items():
#             # random_actions = environment.action_space(agent_id).sample()
#             actions[agent_id] = ppo_agent.compute_single_action(agent_obs)
#         new_state, rewards, done, truncated, infos = env.step(actions)
#         iteration += 1
#         # print(iteration)
#         if iteration % 100 == 1:
#             pass

#     plt.ioff()
#     plt.show()
# ----------------------------------
# import argparse
# import os
# from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
# import ray
# import supersuit as ss
# from PIL import Image
# from ray.rllib.algorithms.ppo import PPO
# from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
# from ray.rllib.models import ModelCatalog
# from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
# from ray.tune.registry import register_env
# from torch import nn

# # from pettingzoo.butterfly import pistonball_v6


# class CNNModelV2(TorchModelV2, nn.Module):
#     def __init__(self, obs_space, act_space, num_outputs, *args, **kwargs):
#         TorchModelV2.__init__(self, obs_space, act_space, num_outputs, *args, **kwargs)
#         nn.Module.__init__(self)
#         self.model = nn.Sequential(
#             nn.Conv2d(3, 32, [8, 8], stride=(4, 4)),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, [4, 4], stride=(2, 2)),
#             nn.ReLU(),
#             nn.Conv2d(64, 64, [3, 3], stride=(1, 1)),
#             nn.ReLU(),
#             nn.Flatten(),
#             (nn.Linear(3136, 512)),
#             nn.ReLU(),
#         )
#         self.policy_fn = nn.Linear(512, num_outputs)
#         self.value_fn = nn.Linear(512, 1)

#     def forward(self, input_dict, state, seq_lens):
#         model_out = self.model(input_dict["obs"].permute(0, 3, 1, 2))
#         self._value_out = self.value_fn(model_out)
#         return self.policy_fn(model_out), state

#     def value_function(self):
#         return self._value_out.flatten()


# os.environ["SDL_VIDEODRIVER"] = "dummy"

# parser = argparse.ArgumentParser(
#     description="Render pretrained policy loaded from checkpoint"
# )
# parser.add_argument(
#     "--checkpoint-path",
#     help="Path to the checkpoint. This path will likely be something like this: `~/ray_results/pistonball_v6/PPO/PPO_pistonball_v6_660ce_00000_0_2021-06-11_12-30-57/checkpoint_000050/checkpoint-50`",
# )

# # args = parser.parse_args()

# # if args.checkpoint_path is None:
# #     print("The following arguments are required: --checkpoint-path")
# #     exit(0)

# checkpoint_path = "/home/qrbao/Downloads/code/code2/2predatorpreygame/FlattenedEnvironment/PPO/PPO_FlattenedEnvironment_10ed7_00000_0_2024-11-25_22-42-08/checkpoint_000000"
# ModelCatalog.register_custom_model("CNNModelV2", CNNModelV2)
# import os
# from PIL import Image
# from ray.rllib.algorithms.ppo import PPO
# import ray

# # 初始化 Ray
# ray.init()

# # 指定模型检查点路径
# checkpoint_path = "/home/qrbao/Downloads/code/code2/2predatorpreygame/FlattenedEnvironment/PPO/PPO_FlattenedEnvironment_10ed7_00000_0_2024-11-25_22-42-08/checkpoint_000000"

# # 加载已注册的环境和模型
# PPOagent = PPO.from_checkpoint(checkpoint_path)

# # 加载注册过的环境
# env = FlattenedEnvironment(
#     num_predators=20, num_prey=30, num_berries=50, species_to_function=species_to_function
# )
# # env.reset()

# # def env_creator():
# #     env = environment1
# #     # env = ss.color_reduction_v0(env, mode="B")
# #     env = ss.dtype_v0(env, "float32")
# #     # env = ss.resize_v1(env, x_size=84, y_size=84)
# #     # env = ss.normalize_obs_v0(env, env_min=0, env_max=1)
# #     # env = ss.frame_stack_v1(env, 3)
# #     return env


# # env = env_creator()
# env_name = "FlattenedEnvironment"
# # register_env(env_name, lambda config: ParallelPettingZooEnv(env_creator()))


# # ray.init()

# # PPOagent = PPO.from_checkpoint(checkpoint_path)

# reward_sum = 0
# frame_list = []
# i = 0
# env.reset()

# for agent in env.agent_iter():
#     observation, reward, termination, truncation, info = env.last()
#     reward_sum += reward
#     if termination or truncation:
#         action = None
#     else:
#         action = PPOagent.compute_single_action(observation)

#     env.step(action)
#     i += 1
#     if i % (len(env.possible_agents) + 1) == 0:
#         img = Image.fromarray(env.render())
#         frame_list.append(img)
# env.close()


# print(reward_sum)
# # frame_list[0].save(
# #     "out.gif", save_all=True, append_images=frame_list[1:], duration=3, loop=0
# # )






# from stable_baselines3 import PPO

# # 创建模型
# model = PPO(
#     policy='MlpPolicy', 
#     env=gym_env, 
#     verbose=1, 
#     n_steps=2048, 
#     batch_size=64, 
#     learning_rate=3e-4
# )

# # 开始训练
# model.learn(total_timesteps=100000)

# # 保存模型
# model.save("ppo_predator_prey")

# # 测试模型
# obs = gym_env.reset()
# for _ in range(1000):
#     action, _ = model.predict(obs)
#     obs, rewards, done, info = gym_env.step(action)
#     gym_env.render()

# from ray.rllib.utils.pre_checks.env import check_env

# check_env(gym_env, config={})


# import os

# import ray
# import supersuit as ss
# from ray import tune
# from ray.rllib.algorithms.ppo import PPOConfig
# from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
# from ray.rllib.models import ModelCatalog
# from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
# from ray.tune.registry import register_env
# from torch import nn
# from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
# import torch
# # from pettingzoo.butterfly import pistonball_v6
# # from pettingzoo.butterfly import knights_archers_zombies_v10

# from torch.distributions import Categorical, Normal
# # env.reset(seed=1)
# class MLPModelV2(TorchModelV2, nn.Module):
#     def __init__(self, obs_space, act_space, num_outputs, *args, **kwargs):
#         TorchModelV2.__init__(self, obs_space, act_space, num_outputs, *args, **kwargs)
#         nn.Module.__init__(self)

#         self.model = nn.Sequential(
#             nn.Linear(obs_space.shape[0], 512),
#             nn.ReLU(),
#             nn.Linear(512, 256),
#             nn.ReLU(),
#             nn.Linear(256, 128),
#             nn.ReLU(),
#         )

#         # Separate output layers for discrete and continuous actions
#         self.discrete_policy_fn = nn.Linear(128, 2)  # For 'makeAChild'
#         self.continuous_policy_fn = nn.Linear(128, 4)  # For 'moveVector' (mean and log_std for x and y)
#         self.value_fn = nn.Linear(128, 1)

#     def forward(self, input_dict, state, seq_lens):
#         model_out = self.model(input_dict["obs"])
#         self._value_out = self.value_fn(model_out)

#         # Output for discrete action ('makeAChild')
#         discrete_action_logits = self.discrete_policy_fn(model_out)

#         # Output for continuous action ('moveVector')
#         continuous_params = self.continuous_policy_fn(model_out)
#         mean_x = continuous_params[:, 0]
#         log_std_x = continuous_params[:, 1]
#         mean_y = continuous_params[:, 2]
#         log_std_y = continuous_params[:, 3]

#         # Create continuous action distribution
#         std_x = torch.exp(log_std_x)
#         std_y = torch.exp(log_std_y)
#         continuous_action_mean = torch.stack([mean_x, mean_y], dim=-1)
#         continuous_action_std = torch.stack([std_x, std_y], dim=-1)
#         # print({
#         #     "makeAChild": discrete_action_logits,
#         #     "moveVector": (continuous_action_mean, continuous_action_std)  # Return mean and std
#         # })

#         return {(continuous_action_mean, continuous_action_std)  # Return mean and std
#         }, state

#     def value_function(self):
#         return self._value_out.flatten()
# from pprint import pprint

# from ray import train, tune

# config = (
#     PPOConfig()
#     .api_stack(
#         enable_rl_module_and_learner=True,
#         enable_env_runner_and_connector_v2=True,
#     )
#     .environment("CartPole-v1")
#     .training(
#         lr=tune.grid_search([0.01, 0.001, 0.0001]),
#     )
# )

# tuner = tune.Tuner(
#     "PPO",
#     param_space=config,
#     run_config=train.RunConfig(
#         stop={"env_runners/episode_return_mean": 150.0},
#     ),
# )

# tuner.fit()
# from ray.tune.registry import register_env
# if __name__ == "__main__":
#     ray.init()

#     env_name = "FlattenedEnvironment"

#     def env_creator(env_config):
#         return FlattenedEnvironment()
#     register_env("FlattenedEnvironment", env_creator)
#     register_env(env_name, lambda config: ParallelPettingZooEnv(env_creator(config)))
#     ModelCatalog.register_custom_model("MLPModelV2", MLPModelV2)

#     config = (
#         PPOConfig()
#         .environment(env=env_name, clip_actions=True)
#         .rollouts(num_env_runners=4, rollout_fragment_length=128)
#         .training(
#             train_batch_size=512,
#             lr=2e-5,
#             gamma=0.99,
#             lambda_=0.9,
#             use_gae=True,
#             clip_param=0.4,
#             grad_clip=None,
#             entropy_coeff=0.1,
#             vf_loss_coeff=0.25,
#             # sgd_minibatch_size=64,
#             num_sgd_iter=10,
#         )
#         .debugging(log_level="ERROR")
#         .framework(framework="torch")
#         .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
#     )

#     tune.run(
#         "PPO",
#         name="PPO",
#         stop={"timesteps_total": 50000 if not os.environ.get("CI") else 500},

#         checkpoint_freq=1,
#         storage_path="~/ray_results/" + env_name,
#         config=config.to_dict(),
#     )



# parallel_api_test(environment)
# environment1 = parallel_to_aec(environment1)
# api_test(environment1, num_cycles=1000, verbose_progress=True)
# print(123)
# api_test(environment, num_cycles=1000, verbose_progress=True)
# observations, infos = environment.reset(seed=42)

# # 检查观察值和信息是否与空间匹配
# for agent in environment.agents:
#     obs_space = environment.observation_space(agent)
#     if not obs_space.contains(observations[agent]):
#         print(f"Observation mismatch for {agent}: {observations[agent]}")
#     else:
#         print(f"Observation matches for {agent}.")
#         break
# def run_random_simulation(env):
#     """运行带GUI的随机模拟。"""
#     obs, info = env.reset()

#     data_storage = {
#         'iterations': [],
#         'predator_counts': [],
#         'prey_counts': [],
#         'total_counts': [],
#         'predator_healths': [],
#         'prey_healths': [],
#         'total_healths': []
#     }

#     plt.figure(figsize=(10, 8))
#     plt.ion()
#     iteration = 0
#     game_continue = False

#     while not game_continue:
#         # 调用render函数显示环境
#         env.render()

#         # 每100步更新统计并打印状态
#         # if iteration % 100 == 0:
#         #     update_and_plot(iteration, env, data_storage)

#         # 随机生成动作
#         actions = {agent: env.action_space(agent).sample() for agent in env.agents}
#         new_state, rewards, done, truncated, infos = env.step(actions)

#         iteration += 1

#     plt.ioff()
#     plt.show()
# def action_random(environment):
#     """
#     随机为每个存活的智能体生成动作。
#     """
#     actions = {}
#     for agent_name in environment.agents:
#         # 检查动作空间是否存在
#         if agent_name in environment.action_spaces:
#             actions[agent_name] = environment.action_space(agent_name).sample()
#         else:
#             # 跳过没有定义动作空间的智能体
#             # print(f"Skipping action generation for agent: {agent_name} (no action space found)")
#             pass
#     return actions
# def validate_step_output(environment, observations, rewards, terminations, truncations, infos):
#     print("Validating step output...\n")

#     # Validate observations
#     for agent, obs in observations.items():
#         obs_space = environment.observation_space(agent)
#         if not obs_space.contains(obs):
#             print(f"Observation for agent {agent} does not match the observation space.")
#             print(f"Expected space: {obs_space}")
#             print(f"Actual observation: {obs}")
#         else:
#             # print(f"Observation for agent {agent} matches the observation space.")
            
#             pass
#     # Validate rewards
#     for agent, reward in rewards.items():
#         if not isinstance(reward, (int, float)):
#             print(f"Reward for agent {agent} is not a valid number: {reward}")
#         else:
#             print(f"Reward for agent {agent} is valid: {reward}")

#     # Validate terminations
#     for agent, terminated in terminations.items():
#         if not isinstance(terminated, bool):
#             print(f"Termination flag for agent {agent} is not a boolean: {terminated}")
#         else:
#             print(f"Termination flag for agent {agent} is valid: {terminated}")

#     # Validate truncations
#     for agent, truncated in truncations.items():
#         if not isinstance(truncated, bool):
#             print(f"Truncation flag for agent {agent} is not a boolean: {truncated}")
#         else:
#             print(f"Truncation flag for agent {agent} is valid: {truncated}")

#     # Validate infos
#     for agent, info in infos.items():
#         if not isinstance(info, dict):
#             print(f"Info for agent {agent} is not a dictionary: {info}")
#         else:
#             print(f"Info for agent {agent} is valid: {info}")

#     print("\nStep output validation complete.")

# def update(ev):
#     """
#     调用 step 函数驱动环境，生成随机动作并更新状态。
#     """
#     # 为每个智能体生成随机动作
#     random_actions = action_random(environment)

#     # 使用 step 函数更新环境
#     observations, rewards, terminations, truncations, infos = environment.step(random_actions)

#     # 调用 render 函数更新 GUI 显示
#     environment.render()

#     # # 检查灭绝条件
#     # if len(environment.preys) == 0:
#     #     print("猎物已灭绝！")
#     #     # environment.close()  # 关闭画布
#     #     environment.plot_data()  # 显示统计图表
#     # elif len(environment.predators) == 0:
#     #     print("捕食者已灭绝！")
#     #     # environment.close()  # 关闭画布
#     #     environment.plot_data()  # 显示统计图表

# # # 创建定时器，每0.03秒更新一次
# timer = Timer(interval=0.016, connect=update, start=True)

# # 显式显示画布
# # canvas.show()

# if __name__ == '__main__':
#     from vispy.app import run
#     run()
#     # environment.plot_data()


