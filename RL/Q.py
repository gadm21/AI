

from utils import *









q_table = get_q_table("q_table.pickle")
episodes_rewards = []

for episode in range(episodes):
    if episode % show_every == 0 and episode != 0 : 
        print("episode:", episode)
        show_env = True
    else: 
        show_env = False
        cv2.destroyAllWindows()

    player = Blob(random.choice(colors["player"]))
    enemy = Blob(random.choice(colors["enemy"]))
    food = Blob(random.choice(colors["food"]))
    blobs = {"player":player, "enemy": enemy, "food": food}
    episode_rewards = 0

    for i in range(200):

        observation = get_observation(blobs)
        choice = get_action(q_table, observation)
        blobs["player"].action(choice = choice)
        reward = get_reward(blobs)
        update_q_table(q_table, blobs, observation, choice, reward)
        
        if reward == enemy_penalty or reward == food_reward:
            break

            
        episode_rewards += reward
        if show_env and allow_show:
            env = render(blobs, SIZE)
            if abs(reward) > 10: 
                print("episode:{} reward:{} epsilon:{}".format(episode, reward, epsilon))
                show_image(env, 100)
                break
            else: show_image(env)

    
    episodes_rewards.append(episode_rewards)
    # epsilon = np.maximum(0.1, epsilon*eps_decay)

moving_average = np.convolve(episodes_rewards, np.ones((show_every))//show_every, mode = "valid")
save_plt(moving_average)
save_q_table(q_table, "q_table.pickle")

