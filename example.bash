# if planning to use this script, run from main project folder (Peptide_Folding_Classifier), else change the paths to whatever path necessary

nohup python3 scripts/svm.py -i ./splitData/ -j ./params/svm_params.json -o ./models/svm.pkl -r ./results/results.csv -m ./results/cms/svmCM.png -a ./results/rocs/svmROC.png >& ./logFiles/SVMnohup.out &

nohup python3 scripts/knn.py -i ./splitData/ -j ./params/knn_params.json -o ./models/knn.pkl -r ./results/results.csv -m ./results/cms/knnCM.png -a ./results/rocs/knnROC.png >& ./logFiles/KNNnohup.out &

nohup python3 scripts/feedforward.py -i splitData -o ./models/DL.pkl -r ./results/results.csv -m ./results/cms/cmDL.png -c ./results/rocs/rocDL.png >& ./logFiles/nohupDL.out &

nohup python3 scripts/feedforward.py -i splitData -o ./models/DLTuned.pkl -r ./results/results.csv -m ./results/cms/cmDLTuned.png -c ./results/rocs/rocDLTuned.png -t ./params/dl_params.json >& ./logFiles/nohupDLTuned.out &
