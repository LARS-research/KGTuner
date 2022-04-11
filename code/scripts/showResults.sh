# This script will show all the experiment records in your folder results/

# wn18rr 
python3 showResults.py -dataset wn18rr    -model ComplEx
python3 showResults.py -dataset wn18rr    -model DistMult
python3 showResults.py -dataset wn18rr    -model RESCAL
python3 showResults.py -dataset wn18rr    -model TransE
python3 showResults.py -dataset wn18rr    -model RotatE
python3 showResults.py -dataset wn18rr    -model TuckER
python3 showResults.py -dataset wn18rr    -model ConvE

# FB15k-237
python3 showResults.py -dataset FB15k_237 -model ComplEx
python3 showResults.py -dataset FB15k_237 -model DistMult
python3 showResults.py -dataset FB15k_237 -model RESCAL
python3 showResults.py -dataset FB15k_237 -model TransE
python3 showResults.py -dataset FB15k_237 -model RotatE
python3 showResults.py -dataset FB15k_237 -model TuckER
python3 showResults.py -dataset FB15k_237 -model ConvE

# ogbl-biokg
python3 showResults.py -dataset ogbl-biokg   -model ComplEx
python3 showResults.py -dataset ogbl-biokg   -model DistMult
python3 showResults.py -dataset ogbl-biokg   -model TransE
python3 showResults.py -dataset ogbl-biokg   -model RotatE
python3 showResults.py -dataset ogbl-biokg   -model AutoSF

# ogbl-wikikg2
python3 showResults.py -dataset ogbl-wikikg2 -model ComplEx
python3 showResults.py -dataset ogbl-wikikg2 -model DistMult
python3 showResults.py -dataset ogbl-wikikg2 -model TransE
python3 showResults.py -dataset ogbl-wikikg2 -model RotatE
python3 showResults.py -dataset ogbl-wikikg2 -model AutoSF
