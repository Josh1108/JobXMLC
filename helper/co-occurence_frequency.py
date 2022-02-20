import csv
import pandas as pd
from collections import defaultdict as ddict
import matplotlib.pyplot as plt
import seaborn as sns
dicti =ddict(int)
df = pd.read_csv('../skill_prediction - prediction (1).csv')
df_freq = pd.read_csv('../skill_freqfile.csv')
dicti_freq = ddict(int)

freq_threshold = 20
def run_head_tail_split():
    for i, item in df.iterrows():
        lis = eval(item["True Labels"])
        for i, item in enumerate(lis):
            for j, x in enumerate(lis):
                if x ==item:
                    continue
                else:
                    if item>x:
                        dicti[(item,x)]+=1
                    else:
                        dicti[(x,item)]+=1


    for i, row in df_freq.iterrows():
        dicti_freq[row[0]] = row[1]

    with open("../dumps/co-occurence-freq.csv",'w') as f:
        writer =csv.writer(f)
        writer.writerow(["skill1","skill2","value","type"])

        for key,value in dicti.items():
            if dicti_freq[key[0]]>20 and dicti_freq[key[1]]>20 :
                typ ="h-h"
            elif dicti_freq[key[0]]<=20 and dicti_freq[key[1]]<=20:
                typ = "l-l"
            elif dicti_freq[key[0]]>20 and dicti_freq[key[1]]<=20:
                typ = "h-l"
            else:
                typ = "l-h"
            
            writer.writerow([key[0],key[1],value,typ])


def freq_plot(df_freq):
    keys = list(df_freq['name'])
    # values = [str(x)for x in list(df_freq['num'])]
    plt.figure(figsize=(30,30))
    swarm_plot = sns.histplot(data=df_freq, x='num',bins=[x*100 for x in range(100)],kde=True)
    sns.histplot(data=df_freq, x='num', bins =[x*10 for x in range(100)])
    # swarm_plot.set_xticklabels(swarm_plot.get_xticklabels(), rotation=90, ha="right")
    fig = swarm_plot.get_figure()
    fig.savefig("freq_plot2.png")

freq_plot(df_freq)
