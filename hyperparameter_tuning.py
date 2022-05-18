import optuna
import json
import os
from train_main import train_optuna
import wandb


STUDY_NAME = 'colingdb-POS-freq_sort-fasttext-GIN-PR'

def shortlist_options(options, trial):
    
    options['num_epochs'] = trial.suggest_int(name='num_epochs', low=30, high=50, step=10)  # 2
    options['num_HN_epochs'] = trial.suggest_int(name='num_HN_epochs', low=1, high=41, step=20)  # 3
    # opt['lr'] = trial.suggest_float(name='lr',low=1e-5, high=1e-3, log=True)#3
    options['attention_lr'] = trial.suggest_categorical(name='attention_lr',
                                                        choices=[1e-5, 1e-4, 1e-3, 2 * 1e-3, 2 * 1e-4])  # 5
    # opt['dlr_factor'] = trial.suggest_float(name='dlr_factor',low=0.1,high=0.9,step=0.1)#10 opt['mpt'] =
    # trial.suggest_categorical('mpt',[0,1])#2 opt['restrict_edges_num'] =trial.suggest_int('restrict_edges_num',-1,
    # high=10,step =3)#3 # if -1 not restricted opt['restrict_edges_head_threshold'] =trial.suggest_int(
    # 'restrict_edges_head_threshold',-1,1000,step=100) # frequency of edges. No need to change this, let's keep it
    # at 20 for now
    options['num_random_samples'] = trial.suggest_int('num_random_samples', 1, 41, 10)  # negative random samples #4
    # opt['random_shuffle_nbrs'] =trial.suggest_categorical('random_shuffle_nbrs',[0,1]) # shuffle or not, 0/1 #2
    options['fanouts'] = trial.suggest_categorical('fanouts',
                                                   ["1,1,1", "2,2,2", "3,3,3", "4,4,4", "5,5,5", "6,6,6", "7,7,7"])  # 7
    options['num_HN_shortlist'] = trial.suggest_int('num_HN_shortlist', 1, 41, step=10)  # hard negative shorlist #4
    # opt['embedding_type']
    # opt['run_type']
    # opt['num_validation'] = trial.suggest_int('num_validation',-1,30000,step=10000)#3
    # ^ points to be taken for testing, keep -1
    # opt['validation_freq'] = trial.suggest_int('num_validation',-1,30000,step=5000)
    # opt['num_shortlist'] = trial.suggest_int('num_shortlist',20,2000,step=400) #5 number of labels to be shortlisted
    # opt['preopton_introduce_edges'] = trial.suggest_int('preopton_introduce_edges',1,7,2) #4
    options['preopton_introduce_edges'] = int(options['fanouts'].split(",")[0])
    # opt['predict_ova']
    # opt['A']
    # opt['B']
    # opt['name'] 
    return options, trial


def objective(trial):

    with open('commandline_args.txt', 'r') as f:
        shortlist_opt = json.load(f)

    model_dir = "{}/models/GraphXMLModel{}_{}".format(shortlist_opt['dataset'], shortlist_opt['run_type'], shortlist_opt['name'])

    if not os.path.exists(model_dir):
        print("Making model dir...")
        os.makedirs(model_dir)

    shortlist_opt, trial = shortlist_options(shortlist_opt, trial)

    shortlist_opt["trial.number"] = trial.number

    wandb.init(project="JobXMLC",
               entity="skill",
               config=shortlist_opt,
               group=STUDY_NAME)
    # # saving model config as this is req in training.
    # with open(os.path.join(model_dir, 'commandline_args.txt'), 'w') as f:
    #     json.dump(shortlist_opt, f, indent=2)


    result_metrics  = train_optuna(model_dir, trial.params,trial)

    return result_metrics['recall_5']


if __name__ == '__main__':

    study = optuna.create_study(direction="maximize", study_name=STUDY_NAME, pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=200)

