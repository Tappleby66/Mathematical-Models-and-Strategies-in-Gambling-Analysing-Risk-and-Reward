import random
import concurrent.futures
from itertools import count
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import seaborn as sns
from math import floor
import numpy as np
from time import sleep
from random import choice
from tabulate import tabulate
def random_strategy2():
    strategy={}
    bet_sizes={}
    for Bet_count in range(0,4):
        bet_sizes[Bet_count] = random.randint(1,25 )
    for ace in range(0,2):
        if ace == 0:
            for player_hand in range(9,17):
                for dealer_hand in range(2,12):
                    strategy[(ace, player_hand, dealer_hand)] = random.choice([0, 1,2])
        else:
            for player_hand in range(12,21):
                for dealer_hand in range(2,12):
                    strategy[(ace, player_hand, dealer_hand)] = random.choice([0, 1,2])
    return strategy,bet_sizes
def Get_Strategy2(player_hand, dealer_up_card, strategy, ace):
    if player_hand >=17 and ace ==0:
        return 0
    if player_hand <=8:
        return 1
    action = strategy.get((ace, player_hand, dealer_up_card))
    return action
def List_of_start_strategy(list_length):
    List=[]
    for i in range(0,list_length):
        new_strategy= random_strategy2()
        List.append(new_strategy)
    return List
def deck():
    value = [2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 11] * 24
    random.shuffle(value)
    return value
def get_card(shoe,count):
    if not shoe:
        shoe[:] = deck()
        count =0
    card = shoe.pop(0)
    if card in [2, 3, 4, 5, 6]:
        count += 1
    elif card in [10, 11]:
        count -= 1

    return card , count
def adjust_for_aces(hand_values, aces):
    while hand_values > 21 and aces:
        hand_values -= 10
        aces -= 1
    return hand_values,aces
def get_true_count(running_count, shoe):
    remaining_cards = len(shoe)
    if remaining_cards == 0:
        return 0

    remaining_decks = remaining_cards / 52
    true_count = floor(running_count / remaining_decks)
    true_count = max(-8, min(8, true_count))
    if true_count <= -4:
        bet_size = 0 # Minimum bet
    elif -3 <= true_count <= 1:
        bet_size = 1 # Small bet
    elif 2 <= true_count <= 4:
        bet_size = 2  # Medium bet
    else:
        bet_size = 3  # Maximum bet
    return bet_size


def sim(strategy_tuple, runs):
    strategy, bet_sizes = strategy_tuple
    shoe = deck()
    starting_money = 1000  # Each player starts with $1000
    money = starting_money
    trials = 1
    count = 0

    while trials < runs and money > 0:  # Stop if player runs out of money
        Black_jack = 0
        Dealer_hand = 0
        Player_hand = 0
        Dealer_up_card = 0
        Player_Aces = 0
        Dealer_Aces = 0
        true_count = get_true_count(count, shoe)

        bet_size = bet_sizes.get(true_count)
        # Don't bet more than you have
        bet_size = min(bet_size, money)
        for i in range(2):
            Player_card, count = get_card(shoe, count)
            if Player_card == 11:
                Player_Aces += 1
            Player_hand += Player_card
            Dealer_card, count = get_card(shoe, count)
            if Dealer_card == 11:
                Dealer_Aces += 1
            Dealer_hand += Dealer_card
            if i == 0:
                Dealer_up_card = Dealer_hand

        Dealer_hand, Dealer_Aces = adjust_for_aces(Dealer_hand, Dealer_Aces)
        Player_hand, Player_Aces = adjust_for_aces(Player_hand, Player_Aces)

        if Player_hand == 21:
            Black_jack = 1
        else:
            while True:
                a = Get_Strategy2(Player_hand, Dealer_up_card, strategy, Player_Aces)
                if a == 1:
                    Player_card, count = get_card(shoe, count)
                    Player_hand += Player_card
                    if Player_card == 11:
                        Player_Aces += 1
                    Player_hand, Player_Aces = adjust_for_aces(Player_hand, Player_Aces)
                if a == 2:
                    # Can't double if you don't have enough money
                    if money >= 2 * bet_size:
                        bet_size = 2 * bet_size
                        Player_card, count = get_card(shoe, count)
                        Player_hand += Player_card
                        if Player_card == 11:
                            Player_Aces += 1
                        Player_hand, Player_Aces = adjust_for_aces(Player_hand, Player_Aces)
                        break
                    else:
                        Player_card, count = get_card(shoe, count)
                        Player_hand += Player_card
                        if Player_card == 11:
                            Player_Aces += 1
                        Player_hand, Player_Aces = adjust_for_aces(Player_hand, Player_Aces)
                elif a == 0:
                    break

                if Player_hand >= 21:
                    break

        if Black_jack == 1:
            money += bet_size * 1.5  # Blackjack pays 3:2
        elif Player_hand > 21:
            money -= bet_size
        else:
            while Dealer_hand < 17:
                new_card, count = get_card(shoe, count)
                if new_card == 11:
                    Dealer_Aces += 1
                Dealer_hand += new_card
                Dealer_hand, Dealer_Aces = adjust_for_aces(Dealer_hand, Dealer_Aces)

            if Dealer_hand > 21:
                money += bet_size
            elif Player_hand < Dealer_hand:
                money -= bet_size
            elif Player_hand > Dealer_hand:
                money += bet_size
            else:
                pass  # Push - no money change


        trials += 1

    # Return both final money and hands played
    return money

def crossover_strategy(parent1, parent2):
    child_strategy = {}
    child_bet_sizes ={}
    bet_keys = list(parent1[1].keys())
    bet_cut = random.randint(0, len(bet_keys) - 1)
    for key in parent1[0].keys():
        # Randomly choose the action from either parent
        child_strategy[key] = random.choice([parent1[0][key], parent2[0][key]])

    for i, key in enumerate(bet_keys):
        child_bet_sizes[key] = parent2[1][key] if i >= bet_cut else parent1[1][key]

    return child_strategy, child_bet_sizes

def tournament_selection(df, num_parents, tournament_size=3):
    selected_parents = []

    for _ in range(num_parents):
        # Randomly choose individuals
        tournament_contestants = df.sample(n=tournament_size)

        # Select the best (highest score)
        best_parent = tournament_contestants.loc[tournament_contestants["Score"].idxmax(), "player"]

        selected_parents.append(best_parent)

    return selected_parents


def breed(df, num_parents, tournament_size=7, elite_fraction=0.05):
    num_elites = int(elite_fraction * len(df))  # Number of elites carry on to next generation without breeding
    elites = df.iloc[:num_elites][["Dict", "Bet_sizes"]].apply(tuple, axis=1).tolist()

    parents = tournament_selection(df, num_parents, tournament_size)
    offspring_list = []

    for i in range(0, len(parents), 2):
        if i + 1 < len(parents):
            parent1 = tuple(df.loc[df["player"] == parents[i], ["Dict", "Bet_sizes"]].values[0])
            parent2 = tuple(df.loc[df["player"] == parents[i + 1], ["Dict", "Bet_sizes"]].values[0])

            offspring = (crossover_strategy(parent1, parent2))
            offspring = mutate_strategy(offspring, mutation_rate=0.005)
            offspring_list.append(offspring)

    return elites + offspring_list[: len(df) - num_elites]
def mutate_strategy(strategy_tuple, mutation_rate):
    strategy, bet_sizes = strategy_tuple
    mutated_strategy = strategy.copy()
    mutated_bet_sizes = bet_sizes.copy()

    for key in mutated_strategy.keys():
        if random.random() < mutation_rate:
            mutated_strategy[key] = random.choice([0, 1])

    for key in mutated_bet_sizes.keys():
        if random.random() < mutation_rate:
            mutated_bet_sizes[key] = random.randint(1, 10)

    return mutated_strategy, mutated_bet_sizes

def scoretable(stratlist,runs):

    df = pd.DataFrame(columns=["player", "Dict","Bet_sizes", "Score"])
    i=0
    for i in range(len(stratlist)):

        strat_tuple = stratlist[i]
        strategy,bet_sizes = strat_tuple
        x=sim(strat_tuple,runs)
        new_data = {
            "player":i,
            "Dict":[strategy],
            "Bet_sizes":[bet_sizes],
            "Score":[x]
        }
        new_df = pd.DataFrame(new_data)
        df = pd.concat([df,new_df], ignore_index=True)
        i=i+1
    df = df.sort_values(by="Score", ascending=False).reset_index(drop=True)
    min_score = df["Score"].min()

    df["Adjusted_Score"] = df["Score"] - min_score
    total_score = df["Adjusted_Score"].sum()
    df["Probability"] = df["Adjusted_Score"]/total_score
    return df


def runall(start_population=20,generations=15,runs=10):
    start_strategy_list=List_of_start_strategy(start_population)
    df = scoretable(start_strategy_list,runs)
    fitness_score=[]
    fitness_score2 = []
    for l in range(0,generations):
        offspring_strat=breed(df,start_population)
        df=scoretable(offspring_strat,runs)
        fitness_score.append(df["Score"].mean())
        fitness_score2.append(df.iloc[0]["Score"])
        print(l)

    return df,fitness_score,fitness_score2

def display_fitness_scores(fitness_scores,fitness_scores2,filename):
    generations = list(range(1,len(fitness_scores)+1))
    plt.figure(figsize=(8, 5))
    plt.plot(generations, fitness_scores, marker='o', linestyle='-', color='b', label="Mean Fitness Score")
    plt.plot(generations, fitness_scores2, marker='s', linestyle='--', color='r', label="Best Fitness Score ")

    plt.xlabel("Generations")
    plt.ylabel("Fitness Score")
    plt.title("Fitness Score Over Generations")
    plt.legend()
    plt.grid()

    plt.savefig(filename, format="pdf", bbox_inches="tight")
    plt.close()

def plot(thing, num, filename="blackjack_strategy.pdf"):
    strategy = thing.iloc[num]["Dict"]

    # Convert to DataFrame
    df = pd.DataFrame([(k[0], k[1], k[2], v) for k, v in strategy.items()],
                      columns=["Ace", "Player Hand", "Dealer Hand", "Decision"])

    # Map the numeric 'Decision' values to characters (S, H, D)
    decision_map = {
        0: "S",  # Stand
        1: "H",  # Hit
        2: "D"   # Double
    }
    df["Decision_Char"] = df["Decision"].map(decision_map)


    # Define the color palette for each decision (S, H, D)
    decision_colors = {"S": "blue", "H": "red", "D": "orange"}  # Customize these colors

    # Separate heatmaps for Ace = 0 and Ace = 1
    for ace_value in [0, 1]:
        df_filtered = df[df["Ace"] == ace_value]  # Filter by Ace value
        df_pivot_num = df_filtered.pivot(index="Player Hand", columns="Dealer Hand", values="Decision")
        df_pivot_char = df_filtered.pivot(index="Player Hand", columns="Dealer Hand", values="Decision_Char")

        # Ensure correct ordering of the heatmap
        plt.figure(figsize=(8, 6))

        # Plot heatmap with numerical values for color mapping
        sns.heatmap(df_pivot_num, annot=False, fmt="d", linewidths=0.5, linecolor="black", cbar=False,
                    cmap=plt.cm.colors.ListedColormap([decision_colors["S"], decision_colors["H"], decision_colors["D"]]))

        # Overlay decision annotations (S, H, D)
        for i in range(len(df_pivot_num)):
            for j in range(len(df_pivot_num.columns)):
                plt.text(j + 0.5, i + 0.5, df_pivot_char.iloc[i, j],
                         ha="center", va="center", color="black", fontsize=10)

        # Title and labels
        plt.title(f"Blackjack Strategy Heatmap (Ace = {ace_value})")
        plt.xlabel("Dealer Hand")
        plt.ylabel("Player Hand")

        # Save to PDF (separate files for each Ace value)
        filename_split = filename.replace(".pdf", f"_ace{ace_value}.pdf")
        plt.savefig(filename_split, format="pdf", bbox_inches="tight")

        # Show plot
x=runall()
plot(x[0],0)
display_fitness_scores(x[1],x[2],"1.pdf")