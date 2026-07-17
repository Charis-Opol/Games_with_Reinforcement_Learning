# FlappyBird-RL --- How to Run

## 1. Open a terminal

Navigate to the project folder.

``` bash
cd path/to/FlappyBird-RL
```

------------------------------------------------------------------------

## 2. (Optional) Create a virtual environment

### Windows

``` bash
python -m venv .venv
.venv\Scripts\activate
```

### macOS / Linux

``` bash
python3 -m venv .venv
source .venv/bin/activate
```

------------------------------------------------------------------------

## 3. Install the required packages

``` bash
pip install pygame numpy matplotlib torch
```

Or, if you have a `requirements.txt` file:

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

## 4. Verify the project structure

    FlappyBird-RL/
    │
    ├── config.py
    ├── bird.py
    ├── pipe.py
    ├── base.py
    ├── game.py
    ├── main.py
    ├── model.py
    ├── replay_memory.py
    ├── agent.py
    ├── train.py
    ├── play.py
    └── models/

Create the `models` folder if it does not exist.

------------------------------------------------------------------------

## 5. Test the game

Run:

``` bash
python main.py
```

You should see:

-   A blue background
-   A yellow bird
-   Moving pipes
-   Scrolling ground
-   Spacebar makes the bird jump

------------------------------------------------------------------------

## 6. Train the AI

Run:

``` bash
python train.py
```

The AI will initially perform poorly while exploring.

Example output:

    Game 0 | Score 0 | Epsilon 1.000
    Game 1 | Score 0 | Epsilon 0.995
    Game 2 | Score 1 | Epsilon 0.990

When the AI achieves a new best score, the model is automatically saved
as:

    models/flappy_model.pth

------------------------------------------------------------------------

## 7. Watch the trained AI

After training:

``` bash
python play.py
```

The trained model will play Flappy Bird automatically.

------------------------------------------------------------------------

# Troubleshooting

## ModuleNotFoundError

Install the missing package:

``` bash
pip install pygame numpy matplotlib torch
```

## Model file not found

Train the model first:

``` bash
python train.py
```

## Training crashes

Copy the complete Python traceback and use it for debugging. The full
error message identifies the exact file and line causing the issue.

------------------------------------------------------------------------

# Recommended Workflow

1.  Install dependencies.
2.  Run `python main.py` to verify the game.
3.  Run `python train.py` to train the agent.
4.  Wait until a model is saved.
5.  Run `python play.py` to watch the trained AI.
