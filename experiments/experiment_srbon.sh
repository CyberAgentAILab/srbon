datasets=('alpacafarm' 'hh-harmless-t' 'hh-helpful-t' 'alpaca' 'hh-helpful' 'hh-harmless')
# datasets=('alpaca' 'hh-helpful' 'hh-harmless')
models=('mistral-7b-sft-beta')

for dataset in "${datasets[@]}"; do
    # proxys=("OpenAssistant/reward-model-deberta-v3-large-v2" "stanfordnlp/SteamSHP-flan-t5-large" "stanfordnlp/SteamSHP-flan-t5-xl" "llm-blender/PairRM" 'RM-Mistral-7B')
    proxys=('RM-Mistral-7B')
    golds=('openbmb/Eurus-RM-7b')

    for model in "${models[@]}"; do
        for proxy in "${proxys[@]}"; do
            for gold in "${golds[@]}"; do
                if [ "$proxy" != "$gold" ]; then
                    if [ "$dataset" != "alpaca" ]; then
                        ninstances=999
                    else
                        ninstances=805
                    fi

                    echo "Running with --model $model --proxy $proxy --gold $gold --dataset $dataset --ninstances $ninstances"
                    python stochastic_rbon/stochastic_rbon.py --model $model --proxy $proxy --gold $gold --dataset $dataset --ninstances $ninstances
                fi
            done
        done
    done
done
