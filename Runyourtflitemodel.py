#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 14:42:43 2020

@author: base
"""

#import tensorflow as tf # Will throw an error on ARM (pollux) as tensroflow is not installed
import tflite_runtime.interpreter as tflite
from pathlib import Path
import numpy as np
import time

modelpath= Path.cwd() / 'model.tflite'

'''
--------------------------------------------------------
# Load Model
'''
try: 
    #interpreter = tf.lite.Interpreter(str(modelpath)) # Auf x86 über TensorFlow
    interpreter = tflite.Interpreter(str(modelpath)) # Auf ARM über tflite_runtime
except ValueError as e:
    print("Error: Modelfile could not be found. Check if you are in the correct workdirectory. Errormessage:  " + str(e))
    import sys
    sys.exit()

interpreter.allocate_tensors()
# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details() 

'''
--------------------------------------------------------

Do some preprocessing
'''
samples = preprocess_input(your_data) # Do your preprosession here

'''
--------------------------------------------------------

Run inference
'''
Zeit=time.time()
interpreter.set_tensor(input_details[0]['index'], samples)
interpreter.invoke()
output_of_your_model = np.ravel(interpreter.get_tensor(output_details[0]['index']))
print (time.time()-Zeit)
