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
    strategy = [random.choice([0, 1]),random.choice([0, 1]),random.choice([0, 1]),random.choice([0, 1])]
    return strategy
def Get_Strategy2(player_hand, dealer_up_card, strategy, ace):
    if player_hand < 10:
        if dealer_up_card < 7:
            action = strategy[0]
        else:
            action = strategy[1]
    else:
        if dealer_up_card < 7:
            action = strategy[2]
        else:
            action = strategy[3]
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
def get_card(shoe):
    if not shoe:
        shoe[:] = deck()
    card = shoe.pop(0)
    return card
def adjust_for_aces(hand_values, aces):
    while hand_values > 21 and aces:
        hand_values -= 10
        aces -= 1
    return hand_values,aces

def sim(strategy,runs):
    shoe=deck()
    money=0
    trials=1
    l=0
    Count = 0
    while trials<runs:
        bet =2
        Black_jack = 0
        Dealer_hand = 0
        Player_hand = 0
        Dealer_up_card = 0
        Player_Aces = 0
        Dealer_Aces = 0
        for i in range(2):
            Player_card  =get_card(shoe)
            if Player_card==11:
                Player_Aces+=1
            Player_hand += Player_card
            Dealer_card  = get_card(shoe)
            if Dealer_card == 11:
                Dealer_Aces+=1
            Dealer_hand += Dealer_card
            if i == 0:
                Dealer_up_card = Dealer_hand
        Dealer_hand ,Dealer_Aces = adjust_for_aces(Dealer_hand,Dealer_Aces)
        Player_hand ,Player_Aces= adjust_for_aces(Player_hand,Player_Aces)
        if Player_hand == 21:
            Black_jack = 1
        else:
            while True:
                a = Get_Strategy2(Player_hand,Dealer_up_card,strategy,Player_Aces)
                if a == 1:
                    Player_card =get_card(shoe)
                    Player_hand += Player_card
                    if Player_card == 11:
                        Player_Aces += 1
                    Player_hand , Player_Aces = adjust_for_aces(Player_hand, Player_Aces)
                if a==2:
                    bet=4
                    Player_card = get_card(shoe)
                    Player_hand += Player_card
                    if Player_card == 11:
                        Player_Aces += 1
                    Player_hand, Player_Aces = adjust_for_aces(Player_hand, Player_Aces)
                    break
                elif a==0:
                    break

                if Player_hand >= 21:
                    break
        if Black_jack == 1:
            money+=3
        elif Player_hand >21:
            money-=bet
        else:
            while Dealer_hand < 17:
                new_card = get_card(shoe)
                if new_card == 11:
                    Dealer_Aces += 1
                Dealer_hand += new_card
                Dealer_hand ,Dealer_Aces= adjust_for_aces(Dealer_hand, Dealer_Aces)

            if Dealer_hand > 21:
                money +=bet
            elif Player_hand < Dealer_hand:
                money -= bet
            elif Player_hand>Dealer_hand:
                money +=bet
            else:
                money +=0


        trials+=1
    return money


def scoretable(stratlist, runs):
    df = pd.DataFrame(columns=["player", "Dict", "Score"])

    for i in range(len(stratlist)):
        strategy = stratlist[i]
        x = sim(strategy, runs)
        new_data = {
            "player": i,
            "Dict": [strategy],
            "Score": [x]
        }
        new_df = pd.DataFrame(new_data)
        df = pd.concat([df, new_df], ignore_index=True)

    df = df.sort_values(by="Score", ascending=False).reset_index(drop=True)

    # Calculate adjusted scores
    min_score = df["Score"].min()
    df["Adjusted_Score"] = df["Score"] - min_score
    total_score = df["Adjusted_Score"].sum()

    # Compute selection probabilities
    df["Probability"] = df["Adjusted_Score"] / total_score

    # Compute expected count
    population_size = len(stratlist)
    df["Expected_Count"] = df["Probability"] * population_size

    # Simulate roulette wheel selection to get Actual Count
    actual_counts = [0] * population_size
    cumulative_probs = df["Probability"].cumsum().tolist()
    for _ in range(population_size):
        r = random.random()
        for i, cp in enumerate(cumulative_probs):
            if r <= cp:
                actual_counts[i] += 1
                break
    df["Actual_Count"] = actual_counts

    print(tabulate(df, headers="keys", tablefmt="pretty", showindex=False))
    return df


scoretable(List_of_start_strategy(4),10)