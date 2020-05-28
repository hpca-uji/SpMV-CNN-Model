# Run hyperparameter optimization
python3 spmv_cnn_hyperas.py 2400000 Time
python3 spmv_cnn_hyperas.py 2400000 Energy

# Train the models
for f in {12..24..4}; do
   for m in Time Energy EPKG EDRAM; do
      python3 spm_cnn_train.py ${f}00000 $m
   done 
done

# Test the models
for f in {12..24..4}; do
   for m in Time Energy EPKG EDRAM; do
      python3 spm_cnn_test.py ${f}00000 $m
   done 
done
