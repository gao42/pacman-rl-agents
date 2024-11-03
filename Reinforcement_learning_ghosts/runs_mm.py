import imp
import os
import csv
import statistics
import matplotlib.pyplot as plt
from argparse import ArgumentParser, ArgumentTypeError

from pacman_module.pacman import runGame
from pacman_module.pacman import runGames
from pacman_module.ghostAgents import GreedyGhost, SmartyGhost, DumbyGhost

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


def load_agent_from_file(filepath):
    class_mod = None
    expected_class = 'PacmanAgent'
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
ghosts["smarty"] = SmartyGhost
ghosts["dumby"] = DumbyGhost

if __name__ == '__main__':
    usage = """
    USAGE:      python run.py <game_options> <agent_options>
    EXAMPLES:   (1) python run.py
                    - plays a game with the human agent
                      in small maze
    """

    parser = ArgumentParser(usage)
    parser.add_argument('--seed', help='RNG seed', type=int, default=1)
    parser.add_argument(
        '--agentfile',
        help='Python file containing a `PacmanAgent` class.',
        default="humanagent.py")
    parser.add_argument(
        '--ghostagent',
        help='Ghost agent available in the `ghostAgents` module.',
        choices=["dumby", "greedy", "smarty"], default="greedy")
    parser.add_argument(
        '--layout',
        help='Maze layout (from layout folder).',
        default="small")
    parser.add_argument(
        '--silentdisplay',
        help="Disable the graphical display of the game.",
        action="store_true")

    args = parser.parse_args()

    if (args.agentfile == "humanagent.py" and args.silentdisplay):
        print("Human agent cannot play without graphical display")
        exit()
    agent = load_agent_from_file(args.agentfile)(args)

    gagt = ghosts[args.ghostagent]
    nghosts = 2
    if (nghosts > 0):
        gagts = [gagt(i + 1) for i in range(nghosts)]
    else:
        gagts = []
    #total_score, total_computation_time, total_expanded_nodes = runGame(
    #    args.layout, agent, gagts, not args.silentdisplay, expout=0)

    #lay = layout.getLayout(layout_name)
    layout_name = layout_thin_borders(args.layout, args.w)
    display = graphicsDisplay.PacmanGraphics(
        1.0, frameTime=0.1) if (not args.silentdisplay) else textDisplay.NullGraphics()
    
    games = []

    games = runGames( lay, agent, gagts, display, numGames=5, record=False,
            numTraining=0, catchExceptions=False, timeout=5)

    scores = [g.state.getScore() for g in games]
    mean_score = statistics.mean(scores)
    std_dev_score = statistics.stdev(scores)
    mean_scores.append(mean_score)

    wins = [game.state.isWin() for game in games]
    win_rate = wins.count(True) / float(len(wins))
    win_rates.append(win_rate*100)

    # Write to CSV file
    with open('game_scores.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([nt, scores, mean_score, std_dev_score, win_rate])

    print("Mean Total Score : " + str(mean_score) + " at " + str(num_training) + " training games")
    print("Standard Deviation of Scores : " + str(std_dev_score))
    print("Win Rate : " + str(win_rate * 100) + "%")
