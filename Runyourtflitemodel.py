#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 14:42:43 2020

@author: base
"""

import tensorflow as tf
from pathlib import Path
import numpy as np
import time

modelpath= Path.cwd() / 'mmodel.tflite'

'''
--------------------------------------------------------
# Load Model
'''
try: 
    interpreter = tf.lite.Interpreter(str(modelpath))   # input()    # To let the user see the error message
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
samples = preprocess_input(files) # Do your preprosession here

'''
--------------------------------------------------------

Run inference
'''
Zeit=time()
interpreter.set_tensor(input_details[0]['index'], samples)
interpreter.invoke()
output_of_your_model = np.ravel(interpreter.get_tensor(output_details[0]['index']))
print ( Zeit-time() )