from game import FlappyBirdGame

game = FlappyBirdGame()

while True:

    running, score = game.play_step()

    if not running:

        print(f"Game Over! Score: {score}")

        break