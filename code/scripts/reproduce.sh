# This script will show run the optimal config for each model on each dataset

# wn18rr
python3 run.py -evaluate -dataset wn18rr -cpu 4 -valid_steps 2500 -max_steps 100000 -model TransE    -evaluate_times 1 -test_batch_size 16 -data_path ./dataset/ -gpu 0
python3 run.py -evaluate -dataset wn18rr -cpu 4 -valid_steps 2500 -max_steps 100000 -model RotatE    -evaluate_times 1 -test_batch_size 16 -data_path ./dataset/ -gpu 0
python3 run.py -evaluate -dataset wn18rr -cpu 4 -valid_steps 2500 -max_steps 100000 -model ComplEx   -evaluate_times 1 -test_batch_size 16 -data_path ./dataset/ -gpu 0
python3 run.py -evaluate -dataset wn18rr -cpu 4 -valid_steps 2500 -max_steps 100000 -model DistMult  -evaluate_times 1 -test_batch_size 16 -data_path ./dataset/ -gpu 0
python3 run.py -evaluate -dataset wn18rr -cpu 4 -valid_steps 2500 -max_steps 100000 -model TuckER    -evaluate_times 1 -test_batch_size 16 -data_path ./dataset/ -gpu 0
python3 run.py -evaluate -dataset wn18rr -cpu 4 -valid_steps 2500 -max_steps 100000 -model ConvE     -evaluate_times 1 -test_batch_size 16 -data_path ./dataset/ -gpu 0
python3 run.py -evaluate -dataset wn18rr -cpu 4 -valid_steps 2500 -max_steps 100000 -model RESCAL    -evaluate_times 1 -test_batch_size 16 -data_path ./dataset/ -gpu 0

# FB15k-237
python3 run.py -evaluate -dataset FB15k_237 -cpu 4 -valid_steps 2500 -max_steps 100000 -model TransE   -evaluate_times 1 -test_batch_size 16 -data_path ./dataset/ -gpu 0 
python3 run.py -evaluate -dataset FB15k_237 -cpu 4 -valid_steps 2500 -max_steps 100000 -model RotatE   -evaluate_times 1 -test_batch_size 16 -data_path ./dataset/ -gpu 0 
python3 run.py -evaluate -dataset FB15k_237 -cpu 4 -valid_steps 2500 -max_steps 100000 -model ComplEx  -evaluate_times 1 -test_batch_size 16 -data_path ./dataset/ -gpu 0 
python3 run.py -evaluate -dataset FB15k_237 -cpu 4 -valid_steps 2500 -max_steps 100000 -model DistMult -evaluate_times 1 -test_batch_size 16 -data_path ./dataset/ -gpu 0 
python3 run.py -evaluate -dataset FB15k_237 -cpu 4 -valid_steps 2500 -max_steps 100000 -model TuckER   -evaluate_times 1 -test_batch_size 16 -data_path ./dataset/ -gpu 0 
python3 run.py -evaluate -dataset FB15k_237 -cpu 4 -valid_steps 2500 -max_steps 100000 -model ConvE    -evaluate_times 1 -test_batch_size 16 -data_path ./dataset/ -gpu 0 
python3 run.py -evaluate -dataset FB15k_237 -cpu 4 -valid_steps 2500 -max_steps 100000 -model RESCAL   -evaluate_times 1 -test_batch_size 16 -data_path ./dataset/ -gpu 0 

# ogbl-biokg
python3 run.py -evaluate -dataset ogbl-biokg   -cpu 4 -valid_steps 2500 -max_steps 200000 -model TransE   -evaluate_times 1 -test_batch_size 16 -data_path ./dataset/ -gpu 0 
python3 run.py -evaluate -dataset ogbl-biokg   -cpu 4 -valid_steps 2500 -max_steps 200000 -model RotatE   -evaluate_times 1 -test_batch_size 16 -data_path ./dataset/ -gpu 0  
python3 run.py -evaluate -dataset ogbl-biokg   -cpu 4 -valid_steps 2500 -max_steps 200000 -model ComplEx  -evaluate_times 1 -test_batch_size 16 -data_path ./dataset/ -gpu 0 
python3 run.py -evaluate -dataset ogbl-biokg   -cpu 4 -valid_steps 2500 -max_steps 200000 -model DistMult -evaluate_times 1 -test_batch_size 16 -data_path ./dataset/ -gpu 0 
python3 run.py -evaluate -dataset ogbl-biokg   -cpu 4 -valid_steps 2500 -max_steps 200000 -model AutoSF   -evaluate_times 1 -test_batch_size 16 -data_path ./dataset/ -gpu 0 

# ogbl-wikikg2
python3 run.py -evaluate -dataset ogbl-wikikg2 -cpu 4 -valid_steps 2500 -max_steps 400000 -model TransE   -evaluate_times 1 -test_batch_size 16 -data_path ./dataset/ -gpu 0 
python3 run.py -evaluate -dataset ogbl-wikikg2 -cpu 4 -valid_steps 2500 -max_steps 400000 -model RotatE   -evaluate_times 1 -test_batch_size 16 -data_path ./dataset/ -gpu 0  
python3 run.py -evaluate -dataset ogbl-wikikg2 -cpu 4 -valid_steps 2500 -max_steps 400000 -model ComplEx  -evaluate_times 1 -test_batch_size 16 -data_path ./dataset/ -gpu 0 
python3 run.py -evaluate -dataset ogbl-wikikg2 -cpu 4 -valid_steps 2500 -max_steps 400000 -model DistMult -evaluate_times 1 -test_batch_size 16 -data_path ./dataset/ -gpu 0 
python3 run.py -evaluate -dataset ogbl-wikikg2 -cpu 4 -valid_steps 2500 -max_steps 400000 -model AutoSF   -evaluate_times 1 -test_batch_size 16 -data_path ./dataset/ -gpu 0 
