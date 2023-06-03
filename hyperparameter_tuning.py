import optuna
import json
import os
from train_main import train_optuna
from optuna.integration.wandb import WeightsAndBiasesCallback
'''
Right now the config file is not getting saved. THis is a problem. can add config = dictionary in kwargs in wandb_kwargs. 
So best way is to take input of which hyperparameters to finetune. Add rest to config. since the others are treated as a metric
they will come out as a graph.
'''
def objective(trial):
    # Recall@5 v
    dicti ={}

    with open('commandline_args.txt', 'r') as f:
        dicti = json.load(f)

    model_dir = "{}/models/GraphXMLModel{}_{}".format(dicti['dataset'], dicti['run_type'],dicti['name'])
    
    if not os.path.exists(model_dir):
        print("Making model dir...")
        os.makedirs(model_dir)

    # dicti['num_epochs'] = trial.suggest_int(name ='num_epochs',low = 30,high=50, step =10) #2
    # dicti['num_HN_epochs'] = trial.suggest_int(name='num_HN_epochs',low = 1, high = 41, step =20) #3
    # dicti['lr'] = trial.suggest_float(name='lr',low=1e-5, high=1e-3, log=True)#3
    # dicti['attention_lr'] = trial.suggest_categorical(name='attention_lr',choices = [1e-5,1e-4,1e-3,2*1e-3,2*1e-4]) #5
    # dicti['dlr_factor'] = trial.suggest_float(name='dlr_factor',low=0.1,high=0.9,step=0.1)#10
    # dicti['mpt'] = trial.suggest_categorical('mpt',[0,1])#2
    # dicti['restrict_edges_num'] =trial.suggest_int('restrict_edges_num',-1,high=10,step =3)#3 # if -1 not restricted
    # dicti['restrict_edges_head_threshold'] =trial.suggest_int('restrict_edges_head_threshold',-1,1000,step=100) # frequency of edges. No need to change this, let's keep it at 20 for now
    # dicti['num_random_samples'] =trial.suggest_int('num_random_samples',1,20,1) # negative random samples #4
    # # dicti['random_shuffle_nbrs'] =trial.suggest_categorical('random_shuffle_nbrs',[0,1]) # shuffle or not, 0/1 #2
    # dicti['fanouts'] =trial.suggest_categorical('fanouts',["5,5,5","7,7,7"]) #7
    # # dicti['num_HN_shortlist'] = trial.suggest_int('num_HN_shortlist',1,20,step=10) # hard negative shorlist #4
    # dicti['num_HN_shortlist'] = dicti['num_random_samples']
    # # dicti['embedding_type']
    # # dicti['run_type']
    # # dicti['num_validation'] = trial.suggest_int('num_validation',-1,30000,step=10000)#3 # points to be taken for testing
    # # dicti['validation_freq'] = trial.suggest_int('num_validation',-1,30000,step=5000)
    # dicti['num_shortlist'] = trial.suggest_int('num_shortlist',1000,1200,step=100) #5 number of labels to be shortlisted 
    # # dicti['prediction_introduce_edges'] = trial.suggest_int('prediction_introduce_edges',1,7,2) #4
    # dicti['prediction_introduce_edges'] = int(dicti['fanouts'].split(",")[0])
    # dicti['predict_ova']
    # dicti['A']
    # dicti['B']
    # dicti['name'] 

    
    with open(os.path.join(model_dir,'commandline_args.txt'), 'w') as f:
        json.dump(dicti, f, indent=2)
    
    recall_5 = train_optuna(model_dir)

    return recall_5

if __name__ =='__main__':

    # os.environ['WANDB_MODE'] = 'offline' # for testing

    with open('commandline_args.txt', 'r') as f:
        dicti = json.load(f)

    # wandb_kwargs = {"entity":'skill',"project": "colingdb-POS-freq_sort-fasttext-GIN-NR","config":dicti}
    # wandbc = WeightsAndBiasesCallback(metric_name="recall@5", wandb_kwargs=wandb_kwargs)
    study = optuna.create_study(direction="maximize")
    # study.optimize(objective, n_trials=400,callbacks=[wandbc])
    study.optimize(objective, n_trials=400)
    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))