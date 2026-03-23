# Welcome to ECE6882 Reinforcement Learning Project 2 Repo


1. Description and tasks of project:
   
      This project used environment from gymnasium (gym) and it has three problems: Car-Racing-v3, LunarLander-v3, and Humanoid.
   CarRacing-v3 and LunarLander-v3 are from Box2D environment while Humanoid is from MuJoCo environment.
      More details can be found in gym library:
      ```
      https://gymnasium.farama.org/
      ```

   For each of the problem, you are required to write an agent to get the optimizaed solutions. In order to be evaluated easily, please use the following template for each 3 problems and finish those functions:
   ```
   #Agent class:
   class xxxAgent:
       #initialize the agent, with input of number of actions
       def __init__(self,n_actions: int,):

       #agent takes action, with input indicating current state
       #important !!! don't change this function name
       def act(self, state: np.ndarray):

       #agent load parameter (you need to implement this function if you use model based framework such as DQN, PPO, etc)
       def load_parameter(self,file):
   ```
   The agent function names should be: CarRace-v3: CarRaceAgent, LunarLander-v3: LunarLanderAgent, Humanoid: HumanoidAgent
   
   Besides the required functions listed above, you can implement other functions inside the agent class as you wish.
   
   **For CarRaceAgent, if you would like to define a wrapper of the environment to better fit into your agent, please finish the following classes in xxx.py:**
   ```
   def make_env(render_mode=None):
   ```

3. Running evaluation script:

   I provided two sample testcases for each problems to help you verify your agent functionalities and improve your agent performance. However, the eventual official scores for each problems will be tested more cases besides these two. In order to make the evaluation fair, please don't change the testcases by yourself.
   To run the evaluation, simply run the following command for each agent. 
   ```
   python evaluation.py
   ```
   It will output the scores for each testcase. 

4. Rubrics for each agent:

   The overal score are composed of the **summation** of original returns from 8 test cases where 2 of them are released for you for reference, other 6 test cases are hidden for final scores. This score will be used for ranking among all teams.

5. Submission of problems:

   For each problem, First, change the file "xxx.py" into your own group name, for example "Henry.py" with the group name "Henry". Next, put both **evaluation.py** and **{groupname}.py** into a folder "{groupname}_project2", then submit that folder into the following google drive link: https://drive.google.com/drive/folders/1eyOR_2JpYFuz4fZX0F0zMmSrPxA-8En4?dmr=1&ec=wgc-drive-%5Bmodule%5D-goto.
   
