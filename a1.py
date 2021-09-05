import numpy as np
import math
import sympy
import os
import sys

debug = True


def encrypt(key_matrix, plain_text):
    key_matrix = np.array(key_matrix)

    plain_text_matrix = []
    for i in range(len(plain_text)):
        plain_text_matrix.append(ord(plain_text[i]) - 65)
    plain_text_matrix = np.array(plain_text_matrix)
    np.transpose(plain_text_matrix)
    # print(plain_text_matrix)
    
    result = key_matrix.dot(plain_text_matrix)
    # print(result)
    # print(result.shape)

    cipher_text = ""
    for i in range(len(key_matrix)):
        cipher_text += chr(result[i] % 26 + 65)
    return cipher_text

def decrypt(key_matrix, cipher_text):
	key_matrix = np.array(key_matrix)

	a1 = np.linalg.inv(np.matrix(key_matrix))
	a2 = np.linalg.det(key_matrix)
	a3 = sympy.mod_inverse(np.linalg.det(key_matrix), 26)

	key_matrix_inv =  a1*a2*a3

	cipher_text_matrix = []
	for i in range(len(cipher_text)):
		cipher_text_matrix.append(ord(cipher_text[i]) - 65)
	cipher_text_matrix = np.array(cipher_text_matrix)

	# print(key_matrix_inv.shape)
	print(cipher_text_matrix.shape)
	result = np.array(np.dot(key_matrix_inv, cipher_text_matrix))
	print(result)
	# result = result.tolist()

	plain_text = ""
	for i in range(len(cipher_text)):
		plain_text += chr(int(round(result[0][i], 0) % 26 + 65))
	return plain_text


# main function 
if __name__ == "__main__":
    
    # Take input form user using command line argument
    # python3 encryption.py <key file> <plain-text file>
    
    if(len(sys.argv) != 3):
        print("# Wrong command line argument use the following format :")
        print("# python3 encryption.py <key file> <plain-text file>")
        exit()
    
    key_loc = sys.argv[1]
    plain_loc = sys.argv[2]
    
    if(debug):
        print("# Location of key file :",key_loc)
        print("# Location of plain-text file :",plain_loc)
    
    # Checking if the path to the key file is correct or not !
    if (os.path.exists(key_loc)):
        key_file = open(key_loc)
    else:
        print("# Key file does not exist !!!")
        exit()
    
    # Checking if the path to the plain-text file is correct or not !
    if(os.path.exists(plain_loc)):
        plain_file = open(palin_loc)
    else:
        print("# Plain-text file does not exist !!!")
        exit()
    
    # Reading the key file
    if(debug):
        print("# Reading the key file")
    
    
    key_matrix = [[3,2],[3,5]]
    plain_text = "AT"

    cipher_text = encrypt(key_matrix, plain_text)     
    # print("Cipher Matrix: \n", result)
    
    print("Cipher Text: \n", cipher_text)

    decrypted_plain_text = decrypt(key_matrix, cipher_text)

    print("Decrypted plain text: ", decrypted_plain_text)