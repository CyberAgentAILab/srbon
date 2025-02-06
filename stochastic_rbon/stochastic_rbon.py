import pandas as pd
import numpy as np
import os
import argparse
from tqdm import tqdm
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import Bounds
import ot 
np.random.seed(0)


# compute wasserstein distance 
def custom_wasserstein_distance(pi_ref, pi, cost_matrix):
    pi_ref = np.array(pi_ref)
    pi = np.array(pi)
    cost_matrix = np.array(cost_matrix)
    pi_ref /= np.sum(pi_ref)
    pi /= np.sum(pi)
    return ot.emd2(pi_ref, pi, cost_matrix)

def objective_function(pi, R, pi_ref, beta, cost_matrix):
    reward_term = np.dot(pi, R)
    wd = custom_wasserstein_distance(pi_ref, pi, cost_matrix)
    
    return -(reward_term - beta * wd)



def compute_scores(dataset, model, proxy, gold, betas, ninstances=4, ncandidates=4):
    if dataset == 'alpaca' or dataset == 'alpacafarm':
        matrix_file = "matrix/{}/{}/".format(dataset, model) + "{:04d}_eps-0.01_topk-00_topp-1.00_sentbert_" + str(ncandidates)
        prob_file = "logprob/{}/{}/".format(dataset, model) + "{:04d}_eps-0.01_topk-00_topp-1.00"
        length_file = "length_token/{}/{}/".format(dataset, model) + "{:04d}_eps-0.01_topk-00_topp-1.00"
        reward_file = "reward/{}/{}/".format(dataset, model) + "{:04d}_eps-0.01_topk-00_topp-1.00_{}"
    else:
        matrix_file = "matrix/{}/{}/".format(dataset, model) + "{:04d}_eps-0.00_topk-00_topp-0.90_sentbert_" + str(ncandidates)
        prob_file = "logprob/{}/{}/".format(dataset, model) + "{:04d}_eps-0.00_topk-00_topp-0.90"
        reward_file = "reward/{}/{}/".format(dataset, model) + "{:04d}_eps-0.00_topk-00_topp-0.90_{}"
        length_file = "length_token/{}/{}/".format(dataset, model) + "{:04d}_eps-0.00_topk-00_topp-0.90"
    if proxy == gold:
        reward_names = [os.path.basename(proxy)]
    else:
        reward_names = [os.path.basename(proxy), os.path.basename(gold)]

    rows = []
    
    proxy_model = os.path.basename(proxy)
    gold_model = os.path.basename(gold)
    for instance_id in tqdm(range(ninstances)):
        matrix_df = pd.DataFrame(np.loadtxt(matrix_file.format(instance_id))[:ncandidates,:ncandidates])
        rewards = []
        for i in range(len(reward_names)):
            reward_df = pd.read_csv(reward_file.format(instance_id, reward_names[i]))[:ncandidates]
            rewards.append(reward_df)
        reward_df = pd.concat(rewards, axis=1)
        length_df = pd.read_csv(length_file.format(instance_id))[:ncandidates]
        df = pd.concat([matrix_df, reward_df], axis=1)
        df.rename(columns={f'{reward_df.columns[0]}': f'{os.path.basename(proxy)}'}, inplace=True)
        df.rename(columns={f'{reward_df.columns[1]}': f'{os.path.basename(gold)}'}, inplace=True)
        logprob_df = pd.read_csv(prob_file.format(instance_id))[:ncandidates]

        # BoN sampling
        bon_idx = df[proxy_model].argmax()
        bon_score = df.iloc[bon_idx][gold_model]
        
        # MBR
        mbr_idx = df[matrix_df.columns].sum(axis=1).argmax()
        mbr_score = df.iloc[mbr_idx][gold_model]

        # RBoN_WD
        wdrbon_scores = []
        for beta in betas:
            wd_bon = (df[proxy_model] + beta * df[matrix_df.columns].mean(axis=1)).argmax()
            wd_bon_score = df.iloc[wd_bon][gold_model]
            wdrbon_scores.append(wd_bon_score)
            

        # RBoN_KL
        klbon_scores = []
        for beta in betas:
            kl_bon = (df[proxy_model] + beta * logprob_df[model]).argmax()
            kl_bon_score = df.iloc[kl_bon][gold_model]
            klbon_scores.append(kl_bon_score)
            
        # RBoN_SKL
        sklbon_scores = []
        for beta in betas:
            logprob_normalized = logprob_df[model] - np.max(logprob_df[model])
            pi = np.exp(logprob_normalized) * np.exp(beta * df[proxy_model])/  (np.exp(logprob_normalized)*np.exp(beta * df[proxy_model])).sum()
            skl_index = np.random.choice(len(pi), p=pi)
            skl_bon_score = df.iloc[skl_index][gold_model]
            sklbon_scores.append(skl_bon_score)
            
        # RBoN_SWD
        swdbon_scores = []
        for beta in betas:
            logprob_normalized = logprob_df[model] - np.max(logprob_df[model])
            pi_ref = np.exp(logprob_normalized)
            pi_initial = np.ones(len(pi_ref)) / len(pi_ref)
            cost_matrix = df[matrix_df.columns]
            
            constraints = [
                {'type': 'eq', 'fun': lambda pi: np.sum(pi) - 1}, 
                {'type': 'ineq', 'fun': lambda pi: pi - 1e-10}]
            
            bounds = Bounds(0, np.inf)
            result = minimize(objective_function, pi_initial, args=(df[proxy_model], pi_ref, beta, cost_matrix),
                            constraints=constraints, method='SLSQP',bounds=bounds, options={'disp': False})
            
            optimal_pi = result.x
            optimal_pi = optimal_pi / np.sum(optimal_pi)
            swd_index = np.random.choice(len(optimal_pi), p=optimal_pi)
            swd_bon_score = df.iloc[swd_index][gold_model]
            swdbon_scores.append(swd_bon_score)
        
        # RBoN_L
        lbon_scores = []
        for beta in betas:
            l_bon = (df[proxy_model] -(beta / length_df[model])).argmax()
            l_bon_score = df.iloc[l_bon][gold_model]
            lbon_scores.append(l_bon_score)

        oracle_idx = df[gold_model].argmax()
        oracle_score = df.iloc[oracle_idx][gold_model]

        # random sampling
        avg_score = df[gold_model].mean()

        row = [bon_score,mbr_score] + wdrbon_scores +klbon_scores+swdbon_scores+sklbon_scores+lbon_scores+[oracle_score,avg_score]
        rows.append(row)
    result_df = pd.DataFrame(rows,
        columns=['BoN','MBR'] + ['WD-BoN-{}'.format(beta) for beta in betas] +['KL-BoN-{}'.format(beta) for beta in betas]+ ['SWD-BoN-{}'.format(beta) for beta in betas]+ ['SKL-BoN-{}'.format(beta) for beta in betas]+['L-BoN'.format(beta) for beta in betas]+['gold','random'])

    result_df.to_csv(f'Compare_Result/{dataset}/{os.path.basename(model)},g-{os.path.basename(gold)},p-{os.path.basename(proxy)}.csv', index=False)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='alpaca')
    parser.add_argument("--model", type=str, default='mistral-7b-sft-beta')
    parser.add_argument("--proxy", type=str, default='OpenAssistant/reward-model-deberta-v3-large-v2')
    parser.add_argument("--gold", type=str, default='openbmb/Eurus-RM-7b')
    parser.add_argument("--ninstances", type=int, default=4)
    parser.add_argument("--ncandidates", type=int, default=4)

    args = parser.parse_args()
    dataset = args.dataset
    model = args.model
    proxy = args.proxy
    gold = args.gold
    ninstances = args.ninstances
    ncandidates = args.ncandidates


    betas = [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0,5.0,10.0,15.0,20.0]

    os.makedirs(f'./Compare_Result/{dataset}', exist_ok=True)
    compute_scores(dataset, model, proxy, gold, betas, ninstances, ncandidates)