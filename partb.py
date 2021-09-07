import numpy as np
import math
import sympy
import os
import sys

debug = True


# main function 
if __name__ == "__main__":
    
    # Take input form user using command line argument
    # python3 partb.py <cipher-text file> <plain-text file>
    
    if(len(sys.argv) != 3):
        print("# Wrong command line argument use the following format :")
        print("# python3 partb.py <cipher-text file> <plain-text file>")
        exit()
    
    plain_loc = sys.argv[2]
    cipher_loc = sys.argv[1]
    
    if(debug):
        print("# Location of key file :",plain_loc)
        print("# Location of cipher-text file :",cipher_loc)
    
    # Checking if the path to the plain-text file is correct or not !
    if (os.path.exists(plain_loc)):
        key_file = open(plain_loc,'r')
    else:
        print("# Plain-text file does not exist !!!")
        exit()
    
    # Checking if the path to the cipher-text file is correct or not !
    if(os.path.exists(cipher_loc)):
        cipher_file = open(cipher_loc,'r')
    else:
        print("# Cipher-text file does not exist !!!")
        exit()
    
    # Reading the cipher-text file
    if(debug):
        print("\n###### Reading the cipher-text file ######")
    
    cipher = ""
    tmp = cipher_file.readline().upper()
    while(tmp):
        
        # Cleaning the text
        for i in range(len(tmp)):
            if(tmp[i]>='A' and tmp[i]<='Z'):
                cipher = cipher+tmp[i]
        
        tmp = cipher_file.readline().upper()
    
    cipher_len = len(cipher)
    
    if(debug):
        print("# The size of the cipher-text is :",cipher_len)
        print("# Printing the palin-text :\n"+cipher)
    
    # Reading the plain-text file
    if(debug):
        print("\n###### Reading the plain-text file ######")
    
    plain = ""
    tmp = plain_file.readline().upper()
    while(tmp):
        
        # Cleaning the text
        for i in range(len(tmp)):
            if(tmp[i]>='A' and tmp[i]<='Z'):
                plain = plain+tmp[i]
        
        tmp = plain_file.readline().upper()
    
    plain_len = len(plain)
    
    if(debug):
        print("# The size of the plain-text is :",plain_len)
        print("# Printing the palin-text :\n"+plain)
        
    
    
    