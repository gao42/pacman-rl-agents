import imp
import os
import csv
import statistics
import matplotlib.pyplot as plt
from argparse import ArgumentParser, ArgumentTypeError

from pacman_module.pacman import runGame
from pacman_module.pacman import runGames
from pacman_module.ghostAgents import\
    GreedyGhost, LeftyGhost, RandyGhost
import numpy as np

from pacman_module import util, layout
from pacman_module import textDisplay, graphicsDisplay


def restricted_float(x):
    x = float(x)
    if x < 0.1 or x > 1.0:
        raise ArgumentTypeError("%r not in range [0.1, 1.0]" % (x,))
    return x


def positive_integer(x):
    x = int(x)
    if x < 0:
        raise ArgumentTypeError("%r is not >= 0" % (x,))
    return x


def layout_thin_borders(layout, thickness):
    if thickness <= 1:
        return layout
    w = thickness-1
    lay = layout.replace(".lay", "")
    with open("pacman_module/layouts/" + lay + ".lay") as f:
        list_lines = f.readlines()
    old_len = len(list_lines)
    for _ in range(w * 2):
        list_lines[0] = '%' + list_lines[0]
        list_lines[-1] = '%' + list_lines[-1]
    for _ in range(w):
        list_lines.insert(0, list_lines[0])
        list_lines.append(list_lines[0])
    for i in range(w+1, len(list_lines) - w-1):
        list_lines[i] = list_lines[i].replace("\n", "")
        for _ in range(w):
            list_lines[i] += '%'
            list_lines[i] = '%' + list_lines[i]
        list_lines[i] += "\n"
    with open("pacman_module/layouts/" + lay + "_thicker.lay", "w+") as f:
        f.writelines(list_lines)
    return lay + "_thicker.lay"


def load_agent_from_file(filepath, class_module):
    class_mod = None
    expected_class = class_module
    mod_name, file_ext = os.path.splitext(os.path.split(filepath)[-1])

    if file_ext.lower() == '.py':
        py_mod = imp.load_source(mod_name, filepath)

    elif file_ext.lower() == '.pyc':
        py_mod = imp.load_compiled(mod_name, filepath)

    if hasattr(py_mod, expected_class):
        class_mod = getattr(py_mod, expected_class)

    return class_mod


ghosts = {}
ghosts["greedy"] = GreedyGhost
ghosts["randy"] = RandyGhost
ghosts["lefty"] = LeftyGhost

if __name__ == '__main__':
    usage = """
    USAGE:      python run.py <game_options> <agent_options>
    EXAMPLES:   (1) python run.py
                    - plays a game with the human agent
                      in small maze
    """

    parser = ArgumentParser(usage)
    parser.add_argument(
        '--seed',
        help='Seed for random number generator',
        type=int,
        default=1)
    parser.add_argument(
        '--agentfile',
        help='Python file containing a `PacmanAgent` class.',
        default="humanagent.py")
    parser.add_argument(
        '--ghostagent',
        help='Ghost agent available in the `ghostAgents` module.',
        choices=[ "greedy", "lefty", "randy"], default="randy")
    parser.add_argument(
        '--layout',
        help='Maze layout (from layout folder).',
        default="small")
    parser.add_argument(
        '--nghosts',
        help='Maximum number of ghosts in a maze.',
        type=int, default=1)
    parser.add_argument(
        '--hiddenghosts',
        help='Whether the ghost is graphically hidden or not.',
        default=False, action="store_true")
    parser.add_argument(
        '--silentdisplay',
        help="Disable the graphical display of the game.",
        action="store_true")
    # Specific to Project III
    parser.add_argument(
        '--bsagentfile',
        help='Python file containing a `BeliefStateAgent` class.',
        default=None)
    parser.add_argument(
        '--w',
        help='Parameter w as specified in instructions for Project Part 3.',
        type=int, default=1)
    parser.add_argument(
        '--p',
        help='Parameter p as specified in instructions for Project Part 3.',
        type=float, default=0.5)

    args = parser.parse_args()

    if (args.agentfile == "humanagent.py" and args.silentdisplay):
        print("Human agent cannot play without graphical display")
        exit()
    agent = load_agent_from_file(args.agentfile, "PacmanAgent")(args)

    # gagt = ghosts[args.ghostagent]
    # nghosts = args.nghosts
    # if (nghosts > 0):
    #     gagts = [gagt(i + 1, args) for i in range(nghosts)]
    # else:
    #     gagts = []

    gagt = ghosts[args.ghostagent]
    if (args.nghosts > 0):
        gagts = [gagt(i + 1) for i in range(args.nghosts)]
    else:
        gagts = []

    #layout = layout_thin_borders(args.layout, args.w)
    layout_name = layout_thin_borders(args.layout, args.w)
    bsagt = None
    if args.bsagentfile is not None:
        bsagt = load_agent_from_file(
            args.bsagentfile, "BeliefStateAgent")(args)

    #total_score, total_computation_time, total_expanded_nodes = runGame(
    #    layout, agent, gagts, bsagt, not args.silentdisplay,
    #    expout=0, hiddenGhosts=args.hiddenghosts)

    lay = layout.getLayout(layout_name)
    display = graphicsDisplay.PacmanGraphics(
        1.0, frameTime=0.1) if (not args.silentdisplay) else textDisplay.NullGraphics()
    
    games = []

    num_games = 30
    num_training = [25,50,100,200,400,800,1600,3200,6400]

    mean_scores = []

    # Write the header to the CSV file once
    with open('game_scores.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Num Training Games', 'Scores', 'Mean Score', 'Standard Deviation'])

    for nt in num_training:
        games = runGames( lay, agent, gagts, display, numGames=(nt+10), record=False,
            numTraining=nt, catchExceptions=False, timeout=5)

        scores = [g.state.getScore() for g in games]
        mean_score = statistics.mean(scores)
        std_dev_score = statistics.stdev(scores)
        mean_scores.append(mean_score)

        # Write to CSV file
        with open('game_scores.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([nt, scores, mean_score, std_dev_score])

        print("Mean Total Score : " + str(mean_score) + " at " + str(num_training) + " training games")
        print("Standard Deviation of Scores : " + str(std_dev_score))

    # Plot mean scores versus number of training games
    plt.figure(figsize=(10, 6))
    plt.plot(num_training, mean_scores, marker='o', linestyle='-', color='b')
    plt.xlabel('Number of Training Games')
    plt.ylabel('Mean Score')
    plt.title('Mean Scores vs. Number of Training Games')
    plt.grid(True)
    plt.savefig('mean_scores_vs_num_training.png')
    plt.show()

