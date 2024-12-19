from ddqn_fns import *
import matplotlib.pyplot as plt
from IPython.display import clear_output

csv_filename= "DDQN_lunar_main.csv"

def optimize_model(memory, policy_net, target_net, optimizer, batch_size=64, gamma=0.99):
    if len(memory) < batch_size:
        return 0
    transitions = random.sample(memory, batch_size)
    batch = list(zip(*transitions))

    states = torch.tensor(batch[0], dtype=torch.float32, device=device)
    actions = torch.tensor(batch[1], dtype=torch.long, device=device)
    rewards = torch.tensor(batch[2], dtype=torch.float32, device=device)
    next_states = torch.tensor(batch[3], dtype=torch.float32, device=device)
    dones = torch.tensor(batch[4], dtype=torch.float32, device=device)

    actions = actions.unsqueeze(-1)

    state_action_values = policy_net(states).gather(1, actions).squeeze(-1)
    next_state_values = target_net(next_states).max(1)[0]
    next_state_values *= (1 - dones)  
    expected_state_action_values = (next_state_values * gamma) + rewards

    loss = nn.functional.smooth_l1_loss(state_action_values, expected_state_action_values)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()

optimizer = optim.Adam(policy_net.parameters())
memory = deque(maxlen=10000)
episodes = 1501
sync_freq = 10
batch_size = 64
 
for episode in range(episodes):
    state, _ = env.reset()
    total_reward = 0
    total_loss = 0  
    count_steps = 0  
    epsilon = get_epsilon(episode)
    while True:
        action, flag = choose_action(state, policy_net, epsilon)
        next_state, reward, done, truncated, info = env.step(action)
        memory.append((list(state), action, reward, list(next_state), int(done or truncated)))
        state = next_state
        total_reward += reward
        loss= optimize_model(memory, policy_net, target_net, optimizer, batch_size)

        total_loss += loss
        count_steps += 1

        step_array=[episode,count_steps,epsilon,reward,loss,action,flag]

        save_to_csv(step_array,"step_"+csv_filename)
        if done or truncated:
            break
 
    if episode % sync_freq == 0:
        target_net.load_state_dict(policy_net.state_dict())

    torch.save(policy_net.state_dict(), f'lunar_lander/models/DDQN_lunarlander_main_final.pth')
    if episode %100 ==0:
        torch.save(policy_net.state_dict(), f'lunar_lander/models/DDQN_lunar_main_{episode}.pth')

    average_loss = total_loss / count_steps if count_steps != 0 else 0
  
    print(f"episode {episode}, Total reward: {total_reward}, Epsilon: {epsilon}, average loss: {average_loss}")
    save_to_csv([episode, total_reward, epsilon, average_loss], csv_filename)
    clear_output(wait=True)
 
env.close()


num_episodes = 5
env_test = gym.make(environment, render_mode="human")
for i in range(num_episodes):
    state, _ = env_test.reset()
    done = False
    truncated = False 
    total_reward = 0
    while not done and not truncated:
        action = choose_action(state, policy_net, epsilon=0)  
        next_state, reward, done, truncated, info = env_test.step(action)
        state = next_state
        total_reward += reward
    print(f"Test Episode {i + 1}: Total Reward = {total_reward}")

env_test.close()
