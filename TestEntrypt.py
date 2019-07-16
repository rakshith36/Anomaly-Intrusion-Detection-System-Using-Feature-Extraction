


from crypto.Cipher import AES

key = '0123456789abcdef'
IV = 16 * '\x00'           # Initialization vector: discussed later
mode = AES.MODE_CBC
encryptor = AES.new(key, mode, IV=IV)

text = 'j' * 64 + 'i' * 128
ciphertext = encryptor.encrypt(text)
'''
f = open('file.csv','r').read()
ciphertext = encrypt('USERPASSWORD',f.encode('utf8')) #this .encode('utf8') is the bit im unsure about
e = open('file.enc','wb') # file.enc doesn't need to exist, python will create it
e.write(ciphertext)
e.close'''