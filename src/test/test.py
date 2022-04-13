from function_class import TestFunctions
from function_class import Function
from autoencoder_class import AutoencoderClass
from sklearn.metrics import mean_absolute_error
from contextlib import contextmanager
import sys, os
import random
import numpy as np

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout
            

def compare(func : Function, orig_data, pred_data):
    y_orig = [func(x) for x in orig_data]
    y_pred = [func(x) for x in pred_data]
    y_error = mean_absolute_error(y_orig, y_pred)
    return y_error
            

if __name__ == "__main__":
    func = TestFunctions.get_func('func_1')
    dim, irr, _, generator, normalizer = func.get_params()
    n = 60000
    rand_samles_count = 200
    rand_data = generator.get_lsh(rand_samles_count, irr)
    norm_data = normalizer.normalize(rand_data)
    with open('../../Saved models/Test/' + 'test.txt', 'w') as f:
        for i in range(2, 8):
            f.write(f'Size 8 to {i}\n')
            for enc_type in ['dense', 'deep', 'vae']:
                with suppress_stdout():
                    b_size = 16
                    tr_size = (int(n * 0.8) // 10) * 10
                    
                    sobol_data = generator.get_sobol(n, irr)
                    random.shuffle(sobol_data)
                    data_train = np.array(sobol_data[0 : tr_size])
                    data_test = np.array(sobol_data[tr_size : n])             
                    
                    if (enc_type == 'vae'):
                        while(tr_size % b_size != 0):
                            b_size -= 1
                    
                    model = AutoencoderClass(func, i, enc_type)
                    model.fit(data_train, data_test, 35, b_size, True)
                    pred_data = normalizer.renormalize([model.predict(np.array(xx).reshape(1,dim))[0] for xx in norm_data])
                    err = compare(func, rand_data, pred_data)
                    f.write(f'{enc_type}: {err}\n')
            f.write('\n')