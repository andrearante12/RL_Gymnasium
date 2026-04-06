import numpy as np
import imageio.v2 as imageio
#your submission file
from Arante_Andre import xxxAgent, make_env


def evaluation(env_id="CarRacing-v3", env=None, agent=None, testcase=[0,2]):
    rets = []
    for ep in range(len(testcase)):
        frames=[]
        obs, info = env.reset(seed=testcase[ep])
        done = False
        ep_ret = 0.0
        while not done:
            frames.append(env.render())
            action = agent.act(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_ret += float(reward)
        print("Test case {}---score:{}".format(ep,ep_ret))
        imageio.mimsave("testcase{}.gif".format(ep), frames, fps=30)
        rets.append(ep_ret)

    env.close()
    return float(np.sum(rets))



if __name__ == "__main__":
    # Define your environment
    #example:
    env = make_env(render_mode="rgb_array")
    #initialize your agent
    #example:
    n_actions = env.action_space.shape[0]  # Box action space (continuous)
    input_shape = env.observation_space.shape
    agent = xxxAgent(input_shape, n_actions)
    #load your agent parameter if you have
    #example:
    agent.load_parameter("xxx.pt")
    #run test case
    testcase=[0,2]
    total= evaluation(env_id="CarRacing-v3", env=env,agent=agent, testcase=testcase)
    print("Final eval:", total)
