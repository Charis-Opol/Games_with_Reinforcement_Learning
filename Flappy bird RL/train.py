from agent import Agent
from game import FlappyBirdGame

agent = Agent()

game = FlappyBirdGame()

best_score = 0

while True:

    state_old = agent.get_state(game)

    action = agent.get_action(state_old)

    reward, done, score = game.play_step(action)

    state_new = agent.get_state(game)

    agent.train_short_memory(

        state_old,

        action,

        reward,

        state_new,

        done

    )

    agent.remember(

        state_old,

        action,

        reward,

        state_new,

        done

    )

    if done:

        print(

            f"Game {agent.games:5d} | "

            f"Score {score:3d} | "

            f"Epsilon {agent.epsilon:.3f}"

        )

        agent.train_long_memory()

        agent.end_game()

        if score > best_score:

            best_score = score

            print(

                "New Best Score:",

                best_score

            )

            agent.model.save()

        game.reset()