base=$(pwd)
exp=$1
echo ${exp:4}
for arg in ${*:2}
do
    echo $arg
    sed -i "1c model_checkpoint_path: \"model-$arg\"" $base/$exp/checkpoint
     CUDA_VISIBLE_DEVICES=5 python main.py --mode=decode --config=config --exp_name=${exp:4} --method_name=EncAttentionDecoder
# config, config_retrieval, config_hier
# CovAttentionDecoder, AttentionDecoder, EncAttentionDecoder, RetrievalAttDecoder, EncAttentionSenDecoder
done